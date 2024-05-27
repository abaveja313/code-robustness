import os
from dataclasses import dataclass
from itertools import chain

import tqdm
from loguru import logger

from canonical import MaxProbInitializer
from inference.dataset_loader import DatasetManager
from inference.models import make_model, VllmDecoder
from inference.stem_evaluator import StemEvaluator
from mutations import CRT, RegisteredTransformation
from mutations.registry import MutationRegistry
from shared.structs import MutatedStem, BenchmarkResult
from shared.gcs_storage_manager import GCSResultStorageManager


def evaluate_problem(
    model: VllmDecoder,
    problem_id: str,
    dataset_manager: DatasetManager,
    canonical_samples: int,
    canonical_batch_size: int,
    scoring_samples: int,
    min_correct_samples: int,
    result_manager: GCSResultStorageManager,
    exclude_mutation_types: list[CRT] = None,
):
    logger.info("Finding canonical solution...")
    initializer = MaxProbInitializer(
        model=model,
        problem_id=problem_id,
        num_samples=canonical_samples,
        batch_size=canonical_batch_size,
        min_correct_samples=min_correct_samples,
        dataset_manager=dataset_manager,
    )

    canonical_solution, log_probs = initializer.canonical_solution()
    mutations: list[RegisteredTransformation] = MutationRegistry.get(
        exclude=exclude_mutation_types
    )
    evaluator = StemEvaluator(
        model=model,
        problem_id=problem_id,
        num_samples=scoring_samples,
        dataset_manager=dataset_manager,
    )

    pbar = tqdm.tqdm(list(chain.from_iterable(mutations)))

    for mutation in pbar:
        pbar.set_description(f"{mutation.__class__.__name__}")
        stems: list[MutatedStem] = mutation().get_transformations(
            current_text=canonical_solution
        )
        if len(stems) == 0:
            logger.info("Skipping mutation as it produced no output")
            continue

        for stem in tqdm.tqdm(stems):
            mutation_result = BenchmarkResult(
                problem_id=problem_id, mutation=mutation.__class__.__name__
            )
            evaluator.compute_log_pass_ratio(stem, mutation_result)
            logger.info("Result: " + str(mutation_result))
            result_manager.add(mutation_result)


def benchmark(
    model_name: str,
    dataset_name: str,
    dataset_mini: bool = True,
    dataset_noextreme: bool = False,
    temperature: float = 0.5,
    canonical_samples: int = 200,
    canonical_batch_size: int = 50,
    scoring_samples: int = 100,
    min_correct_samples: int = 10,
    seed_problems_k: int = 5,
    seed_problem_metric: str = "cyclomatic_complexity",
    seed_problems: list[str] = None,
    exclude_mutation_types: list[CRT] = None,
    gcs_bucket_name: str = "amrit-research-samples",
    gcs_project_name: str = "research",
):
    result_manager = GCSResultStorageManager(
        bucket_name=gcs_bucket_name, project=gcs_project_name
    )
    dataset_manager = DatasetManager(
        dataset=dataset_name, mini=dataset_mini, noextreme=dataset_noextreme
    )
    model = make_model(
        model_name, backend="vllm", temperature=temperature, dataset=dataset_name
    )

    if seed_problems is None:
        logger.info("Calculating Seed Problems...")
        seed_problems = dataset_manager.find_seeds(
            k=seed_problems_k, metric=seed_problem_metric
        )

    for seed_problem in seed_problems:
        logger.info(f"Evaluating problem: {seed_problem}")
        evaluate_problem(
            model=model,
            problem_id=seed_problem,
            dataset_manager=dataset_manager,
            canonical_samples=canonical_samples,
            canonical_batch_size=canonical_batch_size,
            scoring_samples=scoring_samples,
            min_correct_samples=min_correct_samples,
            exclude_mutation_types=exclude_mutation_types,
            result_manager=result_manager,
        )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    benchmark(
        dataset_name="mbpp",
        model_name="deepseek-ai/deepseek-coder-1.3b-instruct",
        temperature=0.5,
        canonical_batch_size=200,
        seed_problems_k=10,
        seed_problem_metric="cyclomatic_complexity",
        dataset_mini=False,
        dataset_noextreme=True,
    )
