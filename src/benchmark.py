import os
import pathlib
import sys
from collections import defaultdict
from concurrent.futures import as_completed
from datetime import timedelta
from itertools import chain
from typing import List, Dict, Tuple

import openai
import tqdm
import typer
from loguru import logger
from pebble import ThreadPool

from canonical import MaxProbInitializer
from canonical.max_prob_initializer import NoPassingSolutionException
from inference.dataset_manager import Dataset, SeedStrategy, DatasetManager
from inference.predict import InferenceEngine
from inference.stem_evaluator import StemEvaluator
from mutations import CRT, RegisteredTransformation
from mutations.registry import MutationRegistry
from shared.gcs_storage_manager import GCSResultStorageManager
from shared.logging_utils import LongMessageHashFilter
from shared.structs import MutatedStem, BenchmarkResult

logger.remove()
# Configure the console logger
# hash_filter = LongMessageHashFilter(
#     min_length=200,
#     max_cache_size=512,
#     ttl=timedelta(minutes=10)
# )
# logger.add(sys.stdout, format="{time} {level} {message} {extra[hash]}", level="INFO", filter=hash_filter)
# logger.add("logs/output.log", format="{time} {level} {message} {extra[hash]}", level="DEBUG", filter=hash_filter)

logger.add(sys.stdout, format="{time} {level} {message}", level="INFO")
logger.add("logs/output.log", format="{time} {level} {message}", level="DEBUG")

app = typer.Typer()


def sample_problem_solutions(
        inference_engine: InferenceEngine,
        model_temps: list[float],
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

    pbar = tqdm.tqdm(all_mutations)

    evaluate_targets: Dict[str, Dict[str, str]] = defaultdict(dict)
    results = {}

    with ThreadPool(max_workers=8, max_tasks=50) as executor:
        futures = []
        future_ident_mapping = {}

        for mid, mutation in enumerate(pbar):
            pbar.set_description(f"{mutation.__name__}")
            stems: list[MutatedStem] = mutation().get_transformations(
                current_text=canonical_solution.code
            )
            if len(stems) == 0:
                logger.info("Skipping mutation as it produced no output")
                continue

            for sid, stem in enumerate(tqdm.tqdm(stems)):
                for tid in model_temps:
                    logger.info("Submitting {}-{}-{}-T{}...", problem_id, mid, sid, tid)
                    ident = f"{problem_id}-{mid}-{sid}-T{tid}"
                    results[ident] = BenchmarkResult(
                        problem_id=problem_id,
                        stem_id=str(sid),
                        mutation_id=str(mid),
                        mutation=mutation.__name__,
                        temp=tid
                    )

                    futures.append(executor.submit(
                        inference_engine.sample_stem_solutions,
                        problem_id=problem_id,
                        stem=stem,
                        result=results[ident],
                        temp=tid,
                        num_samples=scoring_samples
                    ))
                    future_ident_mapping[futures[-1]] = ident

        for future in tqdm.tqdm(as_completed(futures), total=len(futures)):
            try:
                completions = future.result()
                ident = future_ident_mapping[future]
                evaluate_targets[ident]['original'] = completions['original']
                evaluate_targets[ident]['mutated'] = completions['mutated']
            except openai.APITimeoutError:
                raise
            except Exception:
                logger.exception("Error during evaluation")

        # Temporary saving in case things go wrong
    eval_target = {
        'evaluate_targets': evaluate_targets,
        'results': results
    }
    logger.info("Adding Data Pickle for Problem: {}", problem_id)
    result_manager.add_data_pickle(eval_target, problem_id)
    del evaluate_targets
    del results


def sample_solutions(
        model_name: str,
        tokenizer_name: str,
        dataset_name: str,
        inference_server_url: str,
        model_max_new_tokens: int = 1024,
        model_temps: Tuple[float] = (0.2, 0.5, 0.8),
        model_top_p: float = 0.9,
        model_direct_completion: bool = False,
        base_only: bool = True,
        dataset_mini: bool = True,
        dataset_noextreme: bool = False,
        canonical_passing_threshold: float = 0.85,
        canonical_samples: int = 200,
        canonical_batch_size: int = 50,
        scoring_samples: int = 100,
        min_correct_samples: int = 10,
        seed_problems_k: int = 5,
        seed_problem_metric: str = "cyclomatic_complexity",
        seed_problems: List[str] = None,
        exclude_mutation_types: List[str] = None,
        gcs_bucket_name: str = "amrit-research-samples",
        gcs_project_name: str = "research",
        completed: List[str] = None,
):
    result_manager = GCSResultStorageManager(
        model_name=model_name, bucket_name=gcs_bucket_name, project=gcs_project_name
    )
    dataset_manager = DatasetManager(
        dataset=dataset_name, mini=dataset_mini, noextreme=dataset_noextreme
    )

    inference_engine = InferenceEngine(
        model_name=model_name,
        tokenizer=tokenizer_name,
        dataset_manager=dataset_manager,
        sampling_args=dict(
            top_p=model_top_p,
            max_tokens=model_max_new_tokens
        ),
        server_url=inference_server_url,
        direct_completion=model_direct_completion
    )

    if seed_problems is None:
        logger.info("Calculating Seed Problems... w/o {}", completed)
        seed_problems = dataset_manager.find_seeds(
            k=seed_problems_k, metric=seed_problem_metric
        )

    for seed_problem in seed_problems:
        if seed_problem in (completed or []):
            logger.info(f"Skipping problem {seed_problem} as it is already completed")
            continue

        logger.info(f"Evaluating problem: {seed_problem}")
        try:
            sample_problem_solutions(
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
                model_temps=model_temps
            )
        except NoPassingSolutionException:
            logger.exception(
                f"Unable to find passing solutions for problem {seed_problem}"
            )


def evaluate_solutions(
        dataset_name: str,
        model_name: str,
        dataset_mini: bool = True,
        dataset_noextreme: bool = False,
        base_only: bool = False,
        max_workers: int = 32,
        max_tasks: int = 15,
        batch_size: int = 250,
        restart_size: int = 25000,
        gcs_bucket_name: str = "amrit-research-samples",
        gcs_project_name: str = "research",
        service_account_path: pathlib.Path = pathlib.Path("/home/user/service-account.json")
):
    logger.info("Evaluating Solutions...")
    result_manager = GCSResultStorageManager(
        model_name=model_name, bucket_name=gcs_bucket_name, project=gcs_project_name,
        service_account_file=str(service_account_path.absolute())
    )
    dataset_manager = DatasetManager(
        dataset=dataset_name, mini=dataset_mini, noextreme=dataset_noextreme
    )
    for problem_id, eval_target, results in tqdm.tqdm(result_manager.get_data_pickles()):
        evaluator = StemEvaluator(
            dataset_manager=dataset_manager,
            problem_id=problem_id,
            base_only=base_only,
            max_workers=max_workers,
            max_tasks=max_tasks,
            batch_size=batch_size,
            restart_size=restart_size
        )
        logger.info("Evaluating {} results...", len(results))
        evaluator.evaluate(eval_target, results)
        logger.info("Done. Writing results to GCS...")
        result_manager.add_all(results)


@app.command(name="eval")
def cli_evaluate_solutions(
        model_name: str = typer.Argument(..., help="The HF name of the model."),
        dataset_name: str = typer.Option(..., help="The name of the dataset."),
        dataset_mini: bool = typer.Option(
            True, help="Whether to use a mini version of the dataset."
        ),
        dataset_noextreme: bool = typer.Option(
            False, help="Whether to exclude extreme samples from the dataset."
        ),
        base_only: bool = typer.Option(True, help="Whether to evaluate base model only."),
        max_workers: int = typer.Option(32, help="Number of workers."),
        max_tasks: int = typer.Option(15, help="Number of tasks."),
        batch_size: int = typer.Option(250, help="Batch size."),
        restart_size: int = typer.Option(25000, help="Restart size."),
        gcs_bucket_name: str = typer.Option(
            "amrit-research-samples", help="Name of the GCS bucket."
        ),
        gcs_project_name: str = typer.Option("research", help="Name of the GCS project."),
        service_account_path: pathlib.Path = typer.Option(pathlib.Path("/home/user/service-account.json"),
                                                          exists=True,
                                                          help="Path to service account file.")
):
    evaluate_solutions(
        model_name=model_name,
        dataset_name=dataset_name,
        dataset_mini=dataset_mini,
        dataset_noextreme=dataset_noextreme,
        base_only=base_only,
        max_workers=max_workers,
        max_tasks=max_tasks,
        batch_size=batch_size,
        restart_size=restart_size,
        gcs_bucket_name=gcs_bucket_name,
        gcs_project_name=gcs_project_name,
        service_account_path=service_account_path
    )


@app.command(name="sample")
def cli_sample_solutions(
        model_name: str = typer.Argument(..., help="The HF name of the model."),
        model_temps: Tuple[float, float, float] = typer.Option((0.3, 0.5, 0.7), help="Temperatures to evaluate at."),
        model_max_new_tokens: int = typer.Option(
            1024, help="Maximum number of new tokens the model can generate."
        ),
        tokenizer_name: str = typer.Option(None, help="The name of the tokenizer (defaults to model)"),
        direct_completion: bool = typer.Option(False, help="Whether to use direct completion."),
        # Codex used 0.95
        model_top_p: float = typer.Option(
            0.95, help="Top-p sampling parameter for the model.", min=0.0, max=1.0
        ),
        base_only: bool = typer.Option(False, help="Whether to evaluate base model only."),
        inference_server: str = typer.Option("http://localhost:5002", help="Inference server URL."),
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
            0.95, help="Passing threshold for canonical samples.", min=0.0, max=1.0
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
        completed: str = typer.Option((), help="Tuple of completed problem IDs."),
):
    sample_solutions(
        model_name=model_name,
        model_temps=model_temps,
        tokenizer_name=tokenizer_name,
        model_direct_completion=direct_completion,
        dataset_name=dataset_name,
        model_max_new_tokens=model_max_new_tokens,
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
        completed=completed.split(','),
        inference_server_url=inference_server
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    app()
