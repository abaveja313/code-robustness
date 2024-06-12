import time
from collections import Counter, defaultdict
from concurrent.futures import as_completed
from typing import Tuple, Dict

from evalplus.evaluate import check_correctness
from loguru import logger
from pebble import ProcessPool
from tqdm import tqdm

from inference.dataset_manager import DatasetManager
from inference.predict import InferenceEngine
from shared.metrics import pass_at_k
from shared.structs import BenchmarkResult, SolutionType


class StemEvaluator:
    def __init__(
            self,
            dataset_manager: DatasetManager,
            problem_id: str,
            num_samples: int = 100,
            k: Tuple[int, ...] = (1, 2, 3, 5, 10),
            base_only: bool = False,
            max_workers: int = 32,
            max_tasks: int = 15,
            batch_size: int = 250,
            restart_size: int = 25000
    ):
        self.dataset_manager = dataset_manager
        self.num_samples = num_samples
        self.problem_id = problem_id
        self.k = k
        self.base_only = base_only
        self.max_workers = max_workers
        self.max_tasks = max_tasks
        self.batch_size = batch_size
        self.restart_size = restart_size

    def process_future_result(self, future, future_meta_mapping, results, pass_stats):
        result_id, result_type, k = future_meta_mapping[future]
        mutated = result_type == "mutated"

        try:
            eval_results = future.result()
            solution: str = eval_results.pop("solution")

            total = eval_results["base"][1]
            if not self.base_only:
                total += eval_results["plus"][1]

            pass_stats[(result_id, result_type)]['total'] += 1

            if len(total) == 0:
                logger.warning("Solution has invalid syntax :\n{}", solution)
                results[result_id].add_example(solution, SolutionType.BAD_SYNTAX, mutated)
                return

            passed = [i for i in total if i == 1]

            if len(passed) == len(total):
                pass_stats[(result_id, result_type)]['pass'] += 1
                results[result_id].add_example(solution, SolutionType.PASSED, mutated)
            else:
                logger.warning("Solution failed:\n{}", solution)
                results[result_id].add_example(solution, SolutionType.FAILED, mutated)

        except Exception as e:
            logger.exception("Error during evaluation")

    def update_results(self, results):
        for result_id, result in results.items():
            result.pass_at_diff = {
                k: result.pass_at_mutated[k] - result.pass_at_original[k]
                for k in self.k
            }
            result.pass_at_ratio = {
                k: result.pass_at_mutated[k] / (result.pass_at_original[k] + 1e-6)
                for k in self.k
            }
            result.compute_metrics()

            logger.info("Result for {}:\n{}", result_id, result)

    def evaluate(self, solutions: Dict[str, Dict[str, str]], results: Dict[str, BenchmarkResult]):
        futures = []
        future_meta_mapping = {}
        completion_id = Counter()
        n_samples = 0
        remaining = set()
        pass_stats = defaultdict(lambda: {"pass": 0, "total": 0})
        completed_jobs = 0

        executor = ProcessPool(max_workers=self.max_workers, max_tasks=self.max_tasks)
        logger.info("Creating process pool with {} workers and {} tasks", self.max_workers, self.max_tasks)

        try:
            for i, result_id in enumerate(tqdm(solutions.keys())):
                for j, key in enumerate(solutions[result_id]):
                    sequences = solutions[result_id][key]
                    for k, sequence in enumerate(sequences):
                        ident = (result_id, key, k)
                        remaining.add(ident)
                        kwargs = dict(
                            dataset=self.dataset_manager.dataset_name,
                            completion_id=completion_id[ident],
                            problem=self.dataset_manager.get_problem(self.problem_id),
                            solution=sequence,
                            expected_output=self.dataset_manager.get_correct(self.problem_id),
                            fast_check=False,
                            base_only=self.base_only,
                            identifier=ident,
                            min_time_limit=1,
                            gt_time_limit_factor=5.0
                        )
                        futures.append(executor.schedule(check_correctness, kwargs=kwargs))
                        future_meta_mapping[futures[-1]] = ident
                        completion_id[ident] += 1
                        n_samples += 1

                        if len(futures) >= self.batch_size:
                            logger.info("Reached batch size, waiting for completion...")
                            logger.debug("Remaining: {}", remaining)
                            for future in as_completed(futures):
                                self.process_future_result(future, future_meta_mapping, results, pass_stats)
                                remaining.remove(future_meta_mapping[future])
                                completed_jobs += 1

                                if completed_jobs % self.restart_size == 0:
                                    logger.warning("Completed jobs reached restart size... restarting executor")
                                    executor.stop()
                                    executor.join()
                                    executor = ProcessPool(max_workers=self.max_workers, max_tasks=self.max_tasks)

                            futures = []

            for future in as_completed(futures):
                self.process_future_result(future, future_meta_mapping, results, pass_stats)
                remaining.remove(future_meta_mapping[future])

        finally:
            logger.warning("Shutting down executor...")
            executor.stop()
            executor.join()

        for result_id, result_type in pass_stats:
            stats = pass_stats[(result_id, result_type)]
            for k in self.k:
                pass_k = pass_at_k(stats['total'], stats['pass'], k)
                if result_type == 'original':
                    results[result_id].pass_at_original[k] = pass_k
                else:
                    results[result_id].pass_at_mutated[k] = pass_k

        self.update_results(results, pass_stats)
