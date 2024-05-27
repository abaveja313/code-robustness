import time
from typing import Tuple

from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import get_groundtruth, check_correctness

from inference.models import VllmDecoder
from shared.metrics import pass_at_k
from shared.mutated_stem import MutatedStem
from loguru import logger

from inference.dataset_loader import DatasetManager


class StemEvaluator:
    def __init__(
            self,
            model: VllmDecoder,
            dataset_manager: DatasetManager,
            problem_id: str,
            num_samples: int = 100,
            k: Tuple[int, ...] = (1, 5, 10)
    ):
        self.model = model
        self.dataset_manager = dataset_manager
        self.num_samples = num_samples
        self.problem_id = problem_id
        self.k = k

    def compute_log_pass_ratio(self, stem: MutatedStem, epsilon=1e-6):
        logger.info("Computing Log Pass Ratio for Stem: {}", stem)

        original_predictions = self.model.complete_stems(stem.original_stem, do_sample=True,
                                                         num_samples=self.num_samples)

        mutated_predictions = self.model.complete_stems(stem.mutated_stem, do_sample=True,
                                                        num_samples=self.num_samples)

        original_pass_at = self.compute_pass_at(stem.original_stem, original_predictions)
        mutated_pass_at = self.compute_pass_at(stem.mutated_stem, mutated_predictions)

        pass_at_ratios = {}
        for k in self.k:
            # add epsilon to avoid division by zero
            pass_at_ratios[f"pass@{k}"] = mutated_pass_at[k] / (original_pass_at[k] + epsilon)

        logger.info("Pass Ratios: {}", pass_at_ratios)
        return pass_at_ratios

    def compute_pass_at(self, stem: str, solutions: list[str]):
        num_passed = 0
        for solution in solutions:
            logger.debug("Computing pass@k for solution: {}", solution)
            full_solution = f"{stem}\n{solution}"
            eval_results = check_correctness(
                dataset=self.dataset_manager.dataset_name,
                completion_id=time.time_ns(),
                expected_output=self.dataset_manager.get_correct(self.problem_id),
                problem=self.dataset_manager.get_problem(self.problem_id),
                solution=full_solution,
                base_only=False,
                gt_time_limit_factor=45.0,
            )

            logger.debug("Eval Results: {}", eval_results)
            total = eval_results["base"][1] + eval_results["plus"][1]
            passed = [i for i in total if i == 1]

            if len(total) == 0:
                logger.warning("Solution {} is syntactically incorrect", full_solution)
                continue

            if not len(total) == 0 and len(total) == len(passed):
                num_passed += 1

        pass_at = {}
        for k in self.k:
            pass_at[k] = pass_at_k(len(solutions), num_passed, k)

        logger.debug("Stem: {}, Pass@: {}", stem, pass_at)
        return pass_at
