import os
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Tuple, Dict

from evalplus.evaluate import check_correctness
from loguru import logger
from tqdm import tqdm

from inference.dataset_manager import DatasetManager
from inference.predict import InferenceEngine
from shared.metrics import pass_at_k
from shared.structs import BenchmarkResult, SolutionType, MutatedStem


class StemEvaluator:
    def __init__(
            self,
            inference_engine: InferenceEngine,
            dataset_manager: DatasetManager,
            problem_id: str,
            num_samples: int = 100,
            k: Tuple[int, ...] = (1, 5, 10),
            base_only: bool = False,
    ):
        self.inference = inference_engine
        self.dataset_manager = dataset_manager
        self.num_samples = num_samples
        self.problem_id = problem_id
        self.k = k
        self.base_only = base_only

    def generate_sequences(
            self,
            stem: MutatedStem,
            result: BenchmarkResult,
            temp: float
    ):
        logger.info(
            "Completing tests (@T{}) for:\n===========\nOld:\n{}\n\nMutated:\n{}",
            temp,
            stem.original_stem,
            stem.mutated_stem,
        )
        result.add_stem(stem)

        predictions, errors = self.inference.complete_stems(
            stem=stem, num_samples=self.num_samples,
            temperature=temp
        )

        logger.warning("Found {} errors during postprocessing", len(errors))

        for error in errors:
            result.add_example(
                example=error.code,
                solution_type=SolutionType.BAD_PROCESS,
                mutated=error.mutated,
            )

        original_predictions = predictions["original"].get_code()
        mutated_predictions = predictions["mutated"].get_code()
        return dict(
            original=original_predictions,
            mutated=mutated_predictions
        )

    def evaluate(self, solutions: Dict[str, Dict[str, str]], results: Dict[str, BenchmarkResult]):
        # Adapted from https://github.com/evalplus/evalplus/blob/master/evalplus/evaluate.py#L29
        with ProcessPoolExecutor(max_workers=int(os.cpu_count() * 0.75), max_tasks_per_child=250) as executor:
            futures = []
            future_meta_mapping = {}
            completion_id = Counter()
            n_samples = 0
            remaining = set()

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
                        futures.append(executor.submit(check_correctness, **kwargs))
                        future_meta_mapping[futures[-1]] = ident
                        completion_id[ident] += 1
                        n_samples += 1

            assert n_samples == len(remaining), "Missing problems in unfinished"

            def stucking_checker():
                while remaining:
                    last_size = len(remaining)
                    time.sleep(20)
                    if last_size != len(remaining) or len(remaining) == 0:
                        continue
                    # Potential stucking
                    logger.warning("No samples had finished testing in the last 20s")
                    logger.warning(f"{len(remaining)} samples to be tested: {remaining}")

            threading.Thread(target=stucking_checker).start()

            pass_stats = defaultdict(lambda: {"pass": 0, "total": 0})

            for future in tqdm(as_completed(futures), total=n_samples):
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
                        continue

                    passed = [i for i in total if i == 1]

                    if len(passed) == len(total):
                        logger.info("Solution passed:\n{}", solution)
                        pass_stats[(result_id, result_type)]['pass'] += 1
                        results[result_id].add_example(solution, SolutionType.PASSED, mutated)
                    else:
                        logger.warning("Solution failed:{}\n{}", total, solution)
                        results[result_id].add_example(solution, SolutionType.FAILED, mutated)

                except Exception as e:
                    logger.error(
                        "Error processing solution: {}\n{}", solution, e
                    )
                finally:
                    remaining.remove((result_id, result_type, k))

            for result_id, result_type in pass_stats:
                stats = pass_stats[(result_id, result_type)]
                for k in self.k:
                    pass_k = pass_at_k(stats['total'], stats['pass'], k)
                    if result_type == 'original':
                        results[result_id].pass_at_original[k] = pass_k
                    else:
                        results[result_id].pass_at_mutated[k] = pass_k

            for result_id, result in results.items():
                result.pass_at_diff = {
                    k: result.pass_at_mutated[k] - result.pass_at_original[k]
                    for k in self.k
                }
                result.pass_at_ratio = {
                    # Add epsilon to prevent division by 0
                    k: result.pass_at_mutated[k] / (result.pass_at_original[k] + 1e-6)
                    for k in self.k
                }
                result.compute_metrics()

                logger.info("Result for {}:\n{}", result_id, result)