import time
from typing import Tuple

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
        logger.info("Computing Log Pass Ratio for Stem: {}", stem)
        result.add_stem(stem)
        original_predictions = self.model.complete_stems(
            stem.original_stem, do_sample=True, num_samples=self.num_samples
        )

        mutated_predictions = self.model.complete_stems(
            stem.mutated_stem, do_sample=True, num_samples=self.num_samples
        )

        original_predictions_post = postprocess(original_predictions)
        mutated_predictions_post = postprocess(mutated_predictions)

        original_pass_at = self.compute_pass_at(
            stem.original_stem, original_predictions_post, "Original"
        )
        mutated_pass_at = self.compute_pass_at(
            stem.mutated_stem, mutated_predictions_post, "Mutated"
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

    def compute_pass_at(
        self, stem: str, solutions: list[str], result: BenchmarkResult, desc: str
    ):
        num_passed = 0

        for solution in solutions:
            eval_results = check_correctness(
                dataset=self.dataset_manager.dataset_name,
                completion_id=time.time_ns(),
                expected_output=self.dataset_manager.get_correct(self.problem_id),
                problem=self.dataset_manager.get_problem(self.problem_id),
                solution=solution,
                base_only=False,
                gt_time_limit_factor=45.0,
            )
            eval_results.pop("completion_id")
            eval_results.pop("task_id")
            eval_results.pop("_identifier")

            logger.debug("{} Eval Results: {}", desc, eval_results)
            total = eval_results["base"][1] + eval_results["plus"][1]
            passed = [i for i in total if i == 1]

            if len(total) == 0:
                logger.error("Solution {} is syntactically incorrect", solution)
                result.bad_syntax_examples.append(solution)
                continue

            if not len(total) == 0 and len(total) == len(passed):
                num_passed += 1
                result.passed_examples.append(solution)
            else:
                result.failed_examples.append(solution)

        pass_at = {}
        for k in self.k:
            pass_at[k] = pass_at_k(len(solutions), num_passed, k)

        logger.debug("Stem: {}, Pass@: {}", stem, pass_at)
        return pass_at
