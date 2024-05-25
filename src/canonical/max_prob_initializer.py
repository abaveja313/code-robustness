import os
import pickle
import time
from functools import cached_property
from typing import List

import joblib
import numpy as np
import tqdm
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth
from loguru import logger
from vllm import CompletionOutput

from inference.models import VllmDecoder
from shared.program_utils import remove_comments_and_docstrings, normalize_indentation

from loguru import logger

class NoPassingSolutionException(Exception):
    pass


class MaxProbInitializer:
    def __init__(
            self,
            model: VllmDecoder,
            problem_id: str,
            passing_threshold: float = 1.0,
            num_samples: int = 300,
            batch_size: int = 50,
            min_correct_samples: int = 10,
            mini=True,
            noextreme=False,
    ):

        if mini and noextreme:
            raise ValueError("Cannot specify both mini=True and noextreme=True")

        self.dataset_params = dict(mini=mini, noextreme=noextreme)
        self.problem = self.human_eval[problem_id]
        self.batch_size = batch_size
        self.passing_threshold = passing_threshold
        self.num_samples = num_samples
        self.model = model
        self.min_correct_samples = min_correct_samples
        self.cache_dir = "cache"

    @cached_property
    def human_eval(self) -> dict[str, dict]:
        return get_human_eval_plus(**self.dataset_params)

    @cached_property
    def ground_truth(self):
        return get_groundtruth(
            self.human_eval, get_human_eval_plus_hash(**self.dataset_params), []
        )

    def _canonical_solution(self):
        logger.info(f"Finding Canonical Solution for {self.task_id}")
        sequences: List[CompletionOutput] = self.batch_generate_sequences()
        sequences: List[CompletionOutput] = self.postprocess_sequences(sequences)
        solutions = []
        failed_stats = []

        for sequence in tqdm.tqdm(sequences, desc="Evaluating Sequences"):
            full_solution = self.problem["prompt"] + sequence.text
            eval_results = check_correctness(
                dataset="humaneval",
                completion_id=time.time_ns(),
                expected_output=self.ground_truth[self.task_id],
                problem=self.problem,
                solution=full_solution,
                base_only=False,
                gt_time_limit_factor=15.0,
            )

            total = eval_results["base"][1] + eval_results["plus"][1]
            if len(total) == 0:
                logger.warning(
                    "No results were found for a syntactically incorrect solution."
                )
                continue

            passed = [i for i in total if i == 1]
            pass_ratio = float(len(passed)) / float(len(total))
            if pass_ratio >= self.passing_threshold:
                solutions.append((full_solution, sequence.cumulative_logprob))
            else:
                failed_stats.append(pass_ratio)

        if len(solutions) < self.min_correct_samples:
            raise NoPassingSolutionException(
                f"Needed {self.min_correct_samples} correct solutions, but found {len(solutions)}"
            )

        self._print_failure_stats(failed_stats)
        canonical_solution = max(solutions, key=lambda s: s[1])
        logger.info(f"Max Probability Solution: {canonical_solution}")

        return canonical_solution

    def batch_generate_sequences(self):
        sequences = []
        remaining = self.num_samples

        with tqdm.tqdm(total=self.num_samples, desc="Generating sequences") as pbar:
            while remaining > 0:
                to_gen = min(self.batch_size, remaining)
                samples = self.model.codegen(
                    prompt=self.problem["prompt"], do_sample=True, num_samples=to_gen
                )[0].outputs
                sequences.extend(samples)
                pbar.update(to_gen)
                remaining -= to_gen

        return sequences

    def canonical_solution(self):
        cache_file = os.path.join(
            self.cache_dir, f'{self.task_id.replace("/", "_")}.pkl'
        )
        if os.path.exists(cache_file):
            logger.info(f"Loading cached centroid solution for task {self.task_id}")
            return joblib.load(cache_file)

        new_solution = self._canonical_solution()
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, "wb") as sol:
            pickle.dump(new_solution, sol)

        return new_solution

    def postprocess_sequences(self, sequences: list[CompletionOutput]):
        transforms = (
            lambda c: remove_comments_and_docstrings(c, remove_docstrings=True),
            normalize_indentation,
        )
        processed = []
        for sequence in tqdm.tqdm(sequences, desc="Postprocessing Samples"):
            try:
                result = sequence.text
                for transform in transforms:
                    result = transform(result)
                sequence.text = result
                processed.append(sequence)
            except Exception:
                logger.exception("Unable to postprocess sequence")
                logger.warning(f"Solution:\n{sequence}")
                continue
        return processed

    def _print_failure_stats(self, fail_rates: list[float]):
        debug_message = [
            "Failure Rate Stats",
            f"Failure Rate: {round(float(len(fail_rates)) / self.num_samples, 4) * 100}%",
            f"Mean: {np.mean(fail_rates)}",
            f"Median: {np.median(fail_rates)}",
            f"Stddev: {np.std(fail_rates)}",
        ]
        logger.info("\n".join(debug_message))
