import os
import time
from concurrent.futures import ThreadPoolExecutor, wait, TimeoutError
from typing import Tuple, List

from evalplus.evaluate import check_correctness
from loguru import logger

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

    def compute_log_pass_ratio(
        self,
        stem: MutatedStem,
        result: BenchmarkResult,
        excluded_tests: list[int],
        epsilon=1e-6,
    ):
        logger.info(
            "Computing Log Pass Ratio for:\n===========\nOld:\n{}\n\nMutated:\n{}",
            stem.original_stem,
            stem.mutated_stem,
        )
        result.add_stem(stem)

        predictions, errors = self.inference.complete_stems(
            stem=stem, num_samples=self.num_samples
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

        original_pass_at = self.compute_pass_at_threaded(
            stem.original_stem,
            original_predictions,
            result,
            "Original",
            excluded_tests,
            mutated=False,
        )
        mutated_pass_at = self.compute_pass_at_threaded(
            stem.mutated_stem,
            mutated_predictions,
            result,
            "Mutated",
            excluded_tests,
            mutated=True,
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

    def check_correctness_wrapper(self, solution):
        logger.debug("Checking correctness for solution:\n{}", solution)
        eval_results = check_correctness(
            dataset=self.dataset_manager.dataset_name,
            completion_id=time.time_ns(),
            expected_output=self.dataset_manager.get_correct(self.problem_id),
            problem=self.dataset_manager.get_problem(self.problem_id),
            solution=solution,
            base_only=self.base_only,
            gt_time_limit_factor=4.0,
        )
        eval_results["solution"] = solution
        return eval_results

    def compute_pass_at_threaded(
        self,
        stem: str,
        solutions: List[str],
        result: BenchmarkResult,
        desc: str,
        excluded_tests: List[int],
        mutated=False,
        initial_timeout=300,
        retry_timeout_increment=10,
        retries=3,
    ):
        num_passed = 0
        remaining_solutions = solutions

        for attempt in range(retries + 1):  # Including the initial attempt
            if not remaining_solutions:
                break

            with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
                future_solution_mapping = {
                    executor.submit(self.check_correctness_wrapper, solution): solution
                    for solution in remaining_solutions
                }

                try:
                    done, not_done = wait(
                        future_solution_mapping.keys(),
                        timeout=initial_timeout + attempt * retry_timeout_increment,
                    )

                    for future in done:
                        try:
                            eval_results = future.result()
                            solution = eval_results.pop("solution")

                            total = eval_results["base"][1]
                            if not self.base_only:
                                total += eval_results["plus"][1]

                            filtered_total = [
                                total[idx]
                                for idx in range(len(total))
                                if idx not in excluded_tests
                            ]
                            logger.debug(
                                "Removing excluded tests: {} -> {}",
                                excluded_tests,
                                filtered_total,
                            )

                            if len(total) == 0:
                                logger.warning(
                                    "Solution has invalid syntax:\n{}", solution
                                )
                                result.add_example(
                                    solution, SolutionType.BAD_SYNTAX, mutated
                                )
                                continue

                            passed = [i for i in filtered_total if i == 1]

                            if len(passed) == len(filtered_total):
                                logger.info(
                                    "Solution passed:{}\n{}", filtered_total, solution
                                )
                                num_passed += 1
                                result.add_example(
                                    solution, SolutionType.PASSED, mutated
                                )
                            else:
                                logger.warning(
                                    "Solution failed:{}\n{}", filtered_total, solution
                                )
                                result.add_example(
                                    solution, SolutionType.FAILED, mutated
                                )

                        except Exception as e:
                            logger.error(
                                "Error processing solution: {}\n{}", solution, e
                            )

                    remaining_solutions = [
                        future_solution_mapping[future] for future in not_done
                    ]
                    if not not_done:
                        break  # All futures completed successfully within the timeout

                    logger.warning(
                        "Timeout occurred. Retrying {} solutions... ({}/{})",
                        len(not_done),
                        attempt + 1,
                        retries,
                    )

                except TimeoutError:
                    logger.error(
                        "Timeout error occurred on attempt {}. Retrying... ({}/{})",
                        attempt + 1,
                        attempt + 1,
                        retries,
                    )
                    remaining_solutions = [
                        future_solution_mapping[future] for future in not_done
                    ]
                    continue
                finally:
                    for future in not_done:
                        future.cancel()

        pass_at = {}
        for k in self.k:
            pass_at[k] = pass_at_k(len(solutions), num_passed, k)
        logger.debug("Stem:\n {}, Pass@: {}", stem, pass_at)
        return pass_at
