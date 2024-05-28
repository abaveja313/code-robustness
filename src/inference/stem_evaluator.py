import multiprocessing
import threading
import time
from functools import partial
from typing import Tuple
from queue import Queue
from evalplus.evaluate import check_correctness
from loguru import logger

from inference.dataset_loader import DatasetManager
from inference.models import VllmDecoder
from shared.metrics import pass_at_k
from shared.program_utils import postprocess
from shared.structs import BenchmarkResult
from shared.structs import MutatedStem


class StemEvaluator:
    def __init__(
            self,
            model: VllmDecoder,
            dataset_manager: DatasetManager,
            problem_id: str,
            num_samples: int = 100,
            k: Tuple[int, ...] = (1, 5, 10),
    ):
        self.model = model
        self.dataset_manager = dataset_manager
        self.num_samples = num_samples
        self.problem_id = problem_id
        self.k = k

    def compute_log_pass_ratio(
            self, stem: MutatedStem, result: BenchmarkResult, epsilon=1e-6
    ):
        logger.info("Computing Log Pass Ratio for:\n===========\nOld:\n{}\n\nMutated:\n{}",
                    stem.original_stem, stem.mutated_stem)
        result.add_stem(stem)
        original_predictions = self.model.complete_stems(
            stem.original_stem, do_sample=True, num_samples=self.num_samples
        )

        mutated_predictions = self.model.complete_stems(
            stem.mutated_stem, do_sample=True, num_samples=self.num_samples
        )

        original_predictions_post = postprocess(original_predictions, result, mutated=False)
        mutated_predictions_post = postprocess(mutated_predictions, result, mutated=True)

        original_pass_at = self.compute_pass_at_threaded(
            stem.original_stem, original_predictions_post, result, "Original", mutated=False
        )
        mutated_pass_at = self.compute_pass_at_threaded(
            stem.mutated_stem, mutated_predictions_post, result, "Mutated", mutated=True
        )
        result.add_pass_ats(original_pass_at, mutated_pass_at)

        pass_at_ratios = {}
        for k in self.k:
            # add epsilon to avoid division by zero
            pass_at_ratios[f"pass@{k}"] = mutated_pass_at[k] / (
                    original_pass_at[k] + epsilon
            )

        result.pass_at_ratio = pass_at_ratios

        logger.info("Pass Ratios: {}", pass_at_ratios)
        return pass_at_ratios

    def check_correctness_wrapper(self, solution, result_queue):
        eval_results = check_correctness(
            dataset=self.dataset_manager.dataset_name,
            completion_id=time.time_ns(),
            expected_output=self.dataset_manager.get_correct(self.problem_id),
            problem=self.dataset_manager.get_problem(self.problem_id),
            solution=solution,
            base_only=False,
            gt_time_limit_factor=45.0,
        )
        eval_results["solution"] = solution
        result_queue.put(eval_results)

    def compute_pass_at_threaded(self, stem: str, solutions: list[str], result: BenchmarkResult, desc: str,
                                 mutated=False):
        num_threads = threading.active_count()
        result_queue = Queue()

        def worker():
            while True:
                solution = solution_queue.get()
                if solution is None:
                    break
                self.check_correctness_wrapper(solution, result_queue)
                solution_queue.task_done()

        solution_queue = Queue()
        for solution in solutions:
            solution_queue.put(solution)

        threads = []
        for _ in range(num_threads):
            t = threading.Thread(target=worker)
            t.start()
            threads.append(t)

        solution_queue.join()

        for _ in range(num_threads):
            solution_queue.put(None)

        for t in threads:
            t.join()

        num_passed = 0
        i = 0
        while not result_queue.empty():
            i += 1

            eval_results = result_queue.get()
            solution = eval_results.pop("solution")
            eval_results.pop("completion_id")
            eval_results.pop("task_id")
            eval_results.pop("_identifier")
            if i % 50 == 0:
                logger.debug("{} Eval Results: {}", desc, eval_results)

            total = eval_results["base"][1] + eval_results["plus"][1]
            passed = [i for i in total if i == 1]
            if len(total) == 0:
                logger.error("Solution {} is syntactically incorrect", solution)
                if mutated:
                    result.bad_syntax_mutated_examples.append(solution)
                else:
                    result.bad_syntax_original_examples.append(solution)
                continue
            if not len(total) == 0 and len(total) == len(passed):
                num_passed += 1
                if mutated:
                    result.passed_mutated_examples.append(solution)
                else:
                    result.passed_original_examples.append(solution)
            else:
                if mutated:
                    result.failed_mutated_examples.append(solution)
                else:
                    result.failed_original_examples.append(solution)

        pass_at = {}
        for k in self.k:
            pass_at[k] = pass_at_k(len(solutions), num_passed, k)
        logger.debug("Stem:\n {}, Pass@: {}", stem, pass_at)
        return pass_at
