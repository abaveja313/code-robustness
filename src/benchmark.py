import os
import sys
from itertools import chain
from typing import List, Tuple

import tqdm
import typer
from loguru import logger

from canonical import MaxProbInitializer
from canonical.max_prob_initializer import NoPassingSolutionException
from inference.dataset_manager import Dataset, SeedStrategy, DatasetManager
from inference.predict import InferenceEngine
from inference.stem_evaluator import StemEvaluator
from mutations import CRT, RegisteredTransformation
from mutations.registry import MutationRegistry
from shared.gcs_storage_manager import GCSResultStorageManager
from shared.logging_utils import create_problem_logger
from shared.structs import MutatedStem, BenchmarkResult

logger.remove()
# Configure the console logger
logger.add(sys.stdout, level="DEBUG")
logger.add("logs/output.log", level="INFO")

app = typer.Typer()


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
    base_only: bool = False,
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
            base_only=base_only,
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
            base_only=base_only,
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
                    problem_id=problem_id,
                    stem_id=str(sid),
                    mutation_id=str(mid),
                    mutation=mutation.__name__,
                )
                evaluator.compute_log_pass_ratio(
                    stem,
                    mutation_result,
                    excluded_tests=canonical_solution.failed_tests,
                )
                logger.info("Result: " + str(mutation_result))
                mutation_result.compute_metrics()
                result_manager.add(mutation_result)


def benchmark(
    model_name: str,
    model_direct_completion: bool,
    model_temp: float,
    dataset_name: str,
    model_max_new_tokens: int = 1024,
    model_dtype: str = "bfloat16",
    model_top_p: float = 0.9,
    base_only: bool = False,
    dataset_mini: bool = True,
    dataset_noextreme: bool = False,
    canonical_passing_threshold: float = 0.85,
    canonical_samples: int = 200,
    canonical_batch_size: int = 50,
    scoring_samples: int = 200,
    min_correct_samples: int = 10,
    seed_problems_k: int = 5,
    seed_problem_metric: str = "cyclomatic_complexity",
    seed_problems: List[str] = None,
    exclude_mutation_types: List[str] = None,
    gcs_bucket_name: str = "amrit-research-samples",
    gcs_project_name: str = "research",
    completed: Tuple[str, ...] = (),
):
    result_manager = GCSResultStorageManager(
        bucket_name=gcs_bucket_name, project=gcs_project_name
    )
    dataset_manager = DatasetManager(
        dataset=dataset_name, mini=dataset_mini, noextreme=dataset_noextreme
    )

    inference_engine = InferenceEngine(
        model_name=model_name,
        dataset_manager=dataset_manager,
        direct_completion=model_direct_completion,
        dtype=model_dtype,
        trust_remote_code=False,
        temperature=model_temp,
        top_p=model_top_p,
        max_tokens=model_max_new_tokens,
    )

    if seed_problems is None:
        logger.info("Calculating Seed Problems...")
        seed_problems = dataset_manager.find_seeds(
            k=seed_problems_k, metric=seed_problem_metric
        )

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
                base_only=base_only,
            )
        except NoPassingSolutionException:
            logger.exception(
                f"Unable to find passing solutions for problem {seed_problem}"
            )


@app.command()
def cli_benchmark(
    model_name: str = typer.Argument(..., help="The HF name of the model."),
    model_direct_completion: bool = typer.Option(
        True, help="Whether the model uses direct completion."
    ),
    model_temp: float = typer.Option(0.5, help="The temperature for the model."),
    model_max_new_tokens: int = typer.Option(
        1024, help="Maximum number of new tokens the model can generate."
    ),
    model_dtype: str = typer.Option("bfloat16", help="Data type used by the model."),
    model_top_p: float = typer.Option(
        0.9, help="Top-p sampling parameter for the model.", min=0.0, max=1.0
    ),
    base_only: bool = typer.Option(False, help="Whether to evaluate base model only."),
    dataset_name: Dataset = typer.Option(Dataset.MBPP, help="The name of the dataset."),
    dataset_mini: bool = typer.Option(
        True, help="Whether to use a mini version of the dataset."
    ),
    dataset_noextreme: bool = typer.Option(
        False, help="Whether to exclude extreme samples from the dataset."
    ),
    seed_problems: List[str] = typer.Option(None, help="List of seed problems."),
    seed_problems_k: int = typer.Option(5, help="Number of seed problems to find."),
    seed_problem_metric: SeedStrategy = typer.Option(
        SeedStrategy.CYCLOMATIC_COMPLEXITY,
        help="Metric to use for finding seed problems.",
    ),
    canonical_passing_threshold: float = typer.Option(
        0.85, help="Passing threshold for canonical samples.", min=0.0, max=1.0
    ),
    canonical_samples: int = typer.Option(200, help="Number of canonical samples."),
    canonical_batch_size: int = typer.Option(
        50, help="Batch size for canonical samples."
    ),
    canonical_min_correct_samples: int = typer.Option(
        10, help="Minimum number of correct samples."
    ),
    pass_at_samples: int = typer.Option(200, help="Number of scoring samples."),
    exclude_mutation_types: List[str] = typer.Option(
        None, help="List of mutation types to exclude."
    ),
    gcs_bucket_name: str = typer.Option(
        "amrit-research-samples", help="Name of the GCS bucket."
    ),
    gcs_project_name: str = typer.Option("research", help="Name of the GCS project."),
    completed: List[str] = typer.Option((), help="Tuple of completed problem IDs."),
):
    benchmark(
        model_name=model_name,
        model_direct_completion=model_direct_completion,
        model_temp=model_temp,
        dataset_name=dataset_name,
        model_max_new_tokens=model_max_new_tokens,
        model_dtype=model_dtype,
        model_top_p=model_top_p,
        base_only=base_only,
        dataset_mini=dataset_mini,
        dataset_noextreme=dataset_noextreme,
        canonical_passing_threshold=canonical_passing_threshold,
        canonical_samples=canonical_samples,
        canonical_batch_size=canonical_batch_size,
        scoring_samples=pass_at_samples,
        min_correct_samples=canonical_min_correct_samples,
        seed_problems_k=seed_problems_k,
        seed_problem_metric=seed_problem_metric,
        seed_problems=seed_problems,
        exclude_mutation_types=exclude_mutation_types,
        gcs_bucket_name=gcs_bucket_name,
        gcs_project_name=gcs_project_name,
        completed=completed,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    app()
