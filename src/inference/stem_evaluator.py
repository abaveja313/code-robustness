import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from queue import Queue
from typing import Tuple

from evalplus.evaluate import check_correctness
from loguru import logger

from inference.dataset_manager import DatasetManager
from inference.predict import InferenceEngine
from inference.processors import PostprocessingException
from shared.metrics import pass_at_k
from shared.structs import BenchmarkResult, SolutionType
from shared.structs import MutatedStem


class StemEvaluator:
    def __init__(
            self,
            inference_engine: InferenceEngine,
            dataset_manager: DatasetManager,
            problem_id: str,
            num_samples: int = 100,
            k: Tuple[int, ...] = (1, 5, 10),
    ):
        self.inference = inference_engine
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

        predictions, errors = self.inference.complete_stems(
            stem=stem,
            num_samples=self.num_samples
        )

        logger.warning("Found {} errors during postprocessing", len(errors))

        for error in errors:
            result.add_example(
                example=error.code,
                solution_type=SolutionType.BAD_PROCESS,
                mutated=error.mutated
            )

        original_predictions = predictions['original'].get_code()
        mutated_predictions = predictions['mutated'].get_code()

        original_pass_at = self.compute_pass_at_threaded(
            stem.original_stem, original_predictions, result, "Original", mutated=False
        )
        mutated_pass_at = self.compute_pass_at_threaded(
            stem.mutated_stem, mutated_predictions, result, "Mutated", mutated=True
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
        def check_correctness_wrapper(solution):
            result_queue = Queue()
            self.check_correctness_wrapper(solution, result_queue)
            return result_queue.get()

        num_passed = 0
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(check_correctness_wrapper, solution) for solution in solutions]

            for i, future in enumerate(as_completed(futures), start=1):
                eval_results = future.result()
                solution = eval_results.pop("solution")
                logger.debug("{} Eval Results: {}", desc, eval_results)

                total = eval_results["base"][1] + eval_results["plus"][1]
                passed = [i for i in total if i == 1]
                if len(total) == 0:
                    logger.error("Solution {} is syntactically incorrect:\n", solution)
                    result.add_example(solution, SolutionType.BAD_SYNTAX, mutated)
                    continue
                if not len(total) == 0 and len(total) == len(passed):
                    num_passed += 1
                    result.add_example(solution, SolutionType.PASSED, mutated)
                else:
                    result.add_example(solution, SolutionType.FAILED, mutated)

        pass_at = {}
        for k in self.k:
            pass_at[k] = pass_at_k(len(solutions), num_passed, k)
        logger.debug("Stem:\n {}, Pass@: {}", stem, pass_at)
        return pass_at
