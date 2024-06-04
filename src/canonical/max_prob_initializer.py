import os
import pickle
import time

import joblib
import numpy as np
import tqdm
from evalplus.evaluate import check_correctness
from loguru import logger

from inference.dataset_manager import DatasetManager
from inference.predict import InferenceEngine
from shared.structs import BatchSolution

from inference.processors import Processors


class NoPassingSolutionException(Exception):
    pass


class MaxProbInitializer:
    def __init__(
        self,
        inference_engine: InferenceEngine,
        dataset_manager: DatasetManager,
        problem_id: str,
        passing_threshold: float = 1.0,
        num_samples: int = 300,
        batch_size: int = 50,
        min_correct_samples: int = 10,
        base_only: bool = False,
    ):
        self.inference_engine = inference_engine
        self.dataset_manager = dataset_manager
        self.problem_id = problem_id
        self.problem = self.dataset_manager.get_problem(problem_id)
        self.batch_size = batch_size
        self.passing_threshold = passing_threshold
        self.num_samples = num_samples
        self.min_correct_samples = min_correct_samples
        self.cache_dir = ".cache"
        self.base_only = base_only

    def _canonical_solution(self):
        logger.info(f"Finding Canonical Solution for {self.problem_id}")
        batch: BatchSolution = self.batch_generate_sequences()

        passing_solutions = []
        pass_ratios = []
        failed_stats = []

        for solution in tqdm.tqdm(batch.solutions, desc="Evaluating Sequences"):
            logger.debug("Checking Correctness of Solution:\n{}", solution.code)
            eval_results = check_correctness(
                dataset=self.dataset_manager.dataset,
                completion_id=time.time_ns(),
                expected_output=self.dataset_manager.get_correct(self.problem_id),
                problem=self.problem,
                solution=solution.code,
                base_only=self.base_only,
                gt_time_limit_factor=45.0,
            )
            logger.debug("Results: {}", eval_results)

            total = eval_results["base"][1]
            if not self.base_only:
                total += eval_results["plus"][1]

            if len(total) == 0:
                logger.warning(
                    f"No results were found for a syntactically incorrect solution.\n{solution.code}"
                )
                continue

            passed = [i for i in total if i == 1]
            solution.failed_tests = [
                idx for idx in range(len(total)) if total[idx] == 0
            ]
            pass_ratio = float(len(passed)) / float(len(total))

            logger.info("Passing Ratio: {}", pass_ratio)
            if pass_ratio >= self.passing_threshold:
                pass_ratios.append(pass_ratio)
                passing_solutions.append(solution)
            else:
                failed_stats.append(pass_ratio)

        if len(passing_solutions) < self.min_correct_samples:
            raise NoPassingSolutionException(
                f"Needed {self.min_correct_samples} correct solutions, but found {len(passing_solutions)}"
            )

        self._print_stats("Failure", failed_stats)

        canonical_solution = max(passing_solutions, key=lambda sol: sol.probs)
        logger.warning(
            f"Max Probability Solution (Probs={canonical_solution.probs}):\n{canonical_solution.code}"
        )
        self._print_stats("Success", pass_ratios)

        canonical_solution.code = Processors.postprocess_canonical(canonical_solution.code)
        return canonical_solution

    def batch_generate_sequences(self):
        batch_solution = BatchSolution()
        remaining = self.num_samples

        with tqdm.tqdm(total=self.num_samples, desc="Generating sequences") as pbar:
            while remaining > 0:
                to_gen = min(self.batch_size, remaining)
                samples, errors = self.inference_engine.predict_solutions(
                    problem_ids=[self.problem_id], num_samples=self.batch_size
                )
                logger.warning("Found {} errors", len(errors))
                batch_solution.add(samples[self.problem_id])
                pbar.update(to_gen)
                remaining -= to_gen

        return batch_solution

    def canonical_solution(self):
        cache_file = os.path.join(
            self.cache_dir, f'{self.problem_id.replace("/", "_")}.pkl'
        )
        if os.path.exists(cache_file):
            logger.debug(f"Loading cached centroid solution for task {self.problem_id}")
            canonical = joblib.load(cache_file)
            logger.info(f"Canonical Solution:\n{canonical}")
            return canonical

        new_solution = self._canonical_solution()
        os.makedirs(self.cache_dir, exist_ok=True)
        with open(cache_file, "wb") as sol:
            pickle.dump(new_solution, sol)

        return new_solution

    def _print_stats(self, rate_type: str, rates: list[float]):
        debug_message = [
            f"{rate_type} Rate Stats",
            f"{rate_type} Rate: {round(float(len(rates)) / self.num_samples, 4) * 100}%",
            f"Mean: {np.mean(rates)}",
            f"Median: {np.median(rates)}",
            f"Stddev: {np.std(rates)}",
        ]
        logger.info("\n".join(debug_message))
