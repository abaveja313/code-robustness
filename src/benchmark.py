import os
from itertools import chain
import random
import sys

import tqdm
from loguru import logger

from canonical import MaxProbInitializer
from canonical.max_prob_initializer import NoPassingSolutionException
from inference.dataset_manager import DatasetManager

from inference.predict import InferenceEngine
from inference.stem_evaluator import StemEvaluator
from mutations import CRT, RegisteredTransformation
from mutations.registry import MutationRegistry
from shared.gcs_storage_manager import GCSResultStorageManager
from shared.structs import MutatedStem, BenchmarkResult
from shared.logging_utils import create_problem_logger

logger.remove()
# Configure the console logger
logger.add(sys.stdout, level="DEBUG")
logger.add("logs/output.log", level="INFO")


def evaluate_problem(
        inference_engine: InferenceEngine,
        problem_id: str,
        dataset_manager: DatasetManager,
        canonical_samples: int,
        canonical_batch_size: int,
        canonical_passing_threshold: float,
        scoring_samples: int,
        min_correct_samples: int,
        result_manager: GCSResultStorageManager,
        exclude_mutation_types: list[CRT] = None,
):
    # Adding a unique file handler for each problem_id
    with create_problem_logger(problem_id=problem_id):
        logger.info("Finding canonical solution...")
        initializer = MaxProbInitializer(
            inference_engine=inference_engine,
            problem_id=problem_id,
            num_samples=canonical_samples,
            passing_threshold=canonical_passing_threshold,
            batch_size=canonical_batch_size,
            min_correct_samples=min_correct_samples,
            dataset_manager=dataset_manager,
        )

        canonical_solution = initializer.canonical_solution()

        mutations: list[RegisteredTransformation] = MutationRegistry.get(
            exclude=exclude_mutation_types
        )
        all_mutations = list(chain.from_iterable(mutations))
        logger.info(f"Found {len(all_mutations)} available mutations")
        evaluator = StemEvaluator(
            inference_engine=inference_engine,
            problem_id=problem_id,
            num_samples=scoring_samples,
            dataset_manager=dataset_manager,
        )

        pbar = tqdm.tqdm(all_mutations)

        for mid, mutation in enumerate(pbar):
            pbar.set_description(f"{mutation.__name__}")
            stems: list[MutatedStem] = mutation().get_transformations(
                current_text=canonical_solution.code
            )
            if len(stems) == 0:
                logger.info("Skipping mutation as it produced no output")
                continue

            for sid, stem in enumerate(tqdm.tqdm(stems)):
                mutation_result = BenchmarkResult(
                    problem_id=problem_id, stem_id=str(sid), mutation_id=str(mid),
                    mutation=mutation.__name__
                )
                evaluator.compute_log_pass_ratio(stem, mutation_result, excluded_tests=canonical_solution.failed_tests)
                logger.info("Result: " + str(mutation_result))
                mutation_result.compute_metrics()
                result_manager.add(mutation_result)


def benchmark(
        model_name: str,
        model_direct_completion: bool,
        model_temp: float,
        dataset_name: str,
        model_max_new_tokens: int = 1024,
        model_dtype="bfloat16",
        model_top_p: float = 0.9,
        dataset_mini: bool = True,
        dataset_noextreme: bool = False,
        canonical_passing_threshold: float = 0.85,
        canonical_samples: int = 200,
        canonical_batch_size: int = 50,
        scoring_samples: int = 200,
        min_correct_samples: int = 10,
        seed_problems_k: int = 5,
        seed_problem_metric: str = "cyclomatic_complexity",
        seed_problems: list[str] = None,
        exclude_mutation_types: list[CRT] = None,
        gcs_bucket_name: str = "amrit-research-samples",
        gcs_project_name: str = "research",
        completed: tuple[str, ...] = ()
):
    result_manager = GCSResultStorageManager(
        bucket_name=gcs_bucket_name,
        project=gcs_project_name
    )
    dataset_manager = DatasetManager(
        dataset=dataset_name,
        mini=dataset_mini,
        noextreme=dataset_noextreme
    )

    inference_engine = InferenceEngine(
        model_name=model_name,
        dataset_manager=dataset_manager,
        direct_completion=model_direct_completion,
        dtype=model_dtype,
        trust_remote_code=False,
        temperature=model_temp,
        top_p=model_top_p,
        max_tokens=model_max_new_tokens
    )

    if seed_problems is None:
        logger.info("Calculating Seed Problems...")
        seed_problems = dataset_manager.find_seeds(
            k=seed_problems_k, metric=seed_problem_metric
        )
        random.shuffle(seed_problems)

    for seed_problem in seed_problems:
        if seed_problem in completed:
            logger.info(f"Skipping problem {seed_problem} as it is already completed")
            continue

        logger.info(f"Evaluating problem: {seed_problem}")
        try:
            evaluate_problem(
                inference_engine=inference_engine,
                problem_id=seed_problem,
                dataset_manager=dataset_manager,
                canonical_samples=canonical_samples,
                canonical_batch_size=canonical_batch_size,
                canonical_passing_threshold=canonical_passing_threshold,
                scoring_samples=scoring_samples,
                min_correct_samples=min_correct_samples,
                exclude_mutation_types=exclude_mutation_types,
                result_manager=result_manager,
            )
        except NoPassingSolutionException:
            logger.exception(f"Unable to find passing solutions for problem {seed_problem}")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    benchmark(
        dataset_name="mbpp",
        model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        model_direct_completion=False,
        model_temp=0.5,
        canonical_batch_size=200,
        canonical_passing_threshold=0.85,
        seed_problems_k=250,
        seed_problem_metric="cyclomatic_complexity",
        dataset_mini=False,
        dataset_noextreme=True
    )
