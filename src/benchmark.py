import os
import pathlib
import sys
from collections import defaultdict
from itertools import chain
from typing import List, Dict

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
from shared.structs import BenchmarkResult

logger.remove()

logger.add(sys.stdout, format="{time} {level} {message} ", level="INFO")
logger.add("logs/output.log", format="{time} {level} {message}", level="DEBUG")

app = typer.Typer()


class Sampler:
    @staticmethod
    def sample_problem_solutions(
            inference_engine: InferenceEngine,
            model_temps: tuple[float, ...],
            problem_id: str,
            dataset_manager: DatasetManager,
            canonical_samples: int,
            canonical_passing_threshold: float,
            scoring_samples: int,
            min_correct_samples: int,
            result_manager: GCSResultStorageManager,
            exclude_mutation_types: list[CRT] = None,
            base_only: bool = False,
    ):
        """
        Sample original and mutated sequences for a given problem and model temperatures.
        Args:
            inference_engine: inference engine to use for sampling
            model_temps: model temperatures to evaluate at
            problem_id: problem ID to evaluate from MBPP or HumanEval
            dataset_manager: dataset manager instance
            canonical_samples: how many solutions to sample for canonical solution
            canonical_passing_threshold: what percentage of tests need to pass for canonical solution to be accepted
            scoring_samples: how many solutions to sample for original and mutated sequences
            min_correct_samples: minimum correct samples to accept a problem
            result_manager: GCS result manager
            exclude_mutation_types: list of mutation classes to exclude
            base_only: whether to only use base tests rather than plus tests

        Returns:

        """
        logger.info("Finding canonical solution...")
        initializer = MaxProbInitializer(
            inference_engine=inference_engine,
            problem_id=problem_id,
            num_samples=canonical_samples,
            passing_threshold=canonical_passing_threshold,
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

        # Collect all stems and calculate total iterations
        all_stems = []
        for mutation in all_mutations:
            stems = mutation().get_transformations(current_text=canonical_solution.code)
            if stems:
                all_stems.append((mutation, stems))

        total_iterations = sum(len(stems) * len(model_temps) for _, stems in all_stems)
        pbar = tqdm.tqdm(total=total_iterations)

        # Store these in GCS for evaluation later
        evaluate_targets: Dict[str, Dict[str, str]] = defaultdict(dict)
        results = {}

        for mid, (mutation, stems) in enumerate(all_stems):
            for tid in model_temps:
                for sid, stem in enumerate(stems):
                    logger.info("Processing {}-{}-{}-T{}...", problem_id, mid, sid, tid)
                    ident = f"{problem_id}-{mid}-{sid}-T{tid}"
                    results[ident] = BenchmarkResult(
                        problem_id=problem_id,
                        stem_id=str(sid),
                        mutation_id=str(mid),
                        mutation=mutation.__name__,
                        temp=tid,
                    )

                    completions = inference_engine.sample_stem_solutions(
                        stem=stem,
                        result=results[ident],
                        temp=tid,
                        num_samples=scoring_samples,
                    )
                    evaluate_targets[ident]["original"] = completions["original"]
                    evaluate_targets[ident]["mutated"] = completions["mutated"]

                    pbar.update(1)

        pbar.close()

        # Temporary saving in case things go wrong
        eval_target = {"evaluate_targets": evaluate_targets, "results": results}
        logger.info("Adding Data Pickle for Problem: {}", problem_id)
        result_manager.add_data_pickle(eval_target, problem_id)
        del evaluate_targets
        del results

    @staticmethod
    def sample_solutions(
            model_name: str,
            dataset_name: str,
            tokenizer_name: str = None,
            model_max_new_tokens: int = 1024,
            model_temps: tuple[float, ...] = (0.2, 0.5, 0.8),
            model_top_p: float = 0.9,
            model_direct_completion: bool = False,
            base_only: bool = True,
            dataset_mini: bool = True,
            dataset_noextreme: bool = False,
            canonical_passing_threshold: float = 0.85,
            canonical_samples: int = 200,
            scoring_samples: int = 100,
            min_correct_samples: int = 10,
            seed_problems_k: int = 5,
            seed_problem_metric: str = "cyclomatic_complexity",
            seed_problems: List[str] = None,
            exclude_mutation_types: List[str] = None,
            gcs_bucket_name: str = "amrit-research-samples",
            gcs_project_name: str = "research",
            completed: List[str] = None,
            service_account_path: pathlib.Path = pathlib.Path(
                "/home/user/service-account.json"
            ),
    ):
        """
        Sample original and mutated sequences for a given model and dataset.

        Args:
            model_name: which model to sample
            tokenizer_name: which tokenizer to use (defaults to model)
            dataset_name: which dataset to use (MBPP/HumanEval)
            model_max_new_tokens: maximum number of tokens to sample for completions
            model_temps: model temperatures to evaluate at
            model_top_p: top p threshold
            model_direct_completion: whether to use a chat template or complete directly
            base_only: whether to only use base tests rather than plus tests
            dataset_mini: whether to use evalplus mini dataset (only if base_only is set)
            dataset_noextreme: whether to exclude extreme samples from the dataset (only if base_only is set)
            canonical_passing_threshold: passing threshold for canonical samples
            canonical_samples: number of canonical samples to consider
            scoring_samples: number of samples to evaluate for original and mutated stems
            min_correct_samples: minimum number of correct samples for canonical solutions
            seed_problems_k: number of seed problems to consider
            seed_problem_metric: which metric to use for ordering seed problems
            seed_problems: explicitly specify seed problems
            exclude_mutation_types: which mutation types to exclude
            gcs_bucket_name: name of the GCS bucket
            gcs_project_name: name of the GCS project
            completed: list of completed problems
            service_account_path: path to service account file
        """
        result_manager = GCSResultStorageManager(
            model_name=model_name,
            bucket_name=gcs_bucket_name,
            project=gcs_project_name,
            service_account_file=str(service_account_path.absolute()),
        )
        dataset_manager = DatasetManager(
            dataset=dataset_name,
            mini=dataset_mini,
            noextreme=dataset_noextreme,
            direct_completion=model_direct_completion,
        )

        inference_engine = InferenceEngine(
            model_name=model_name,
            max_tokens=model_max_new_tokens,
            tokenizer=tokenizer_name,
            dataset_manager=dataset_manager,
            top_p=model_top_p,
            direct_completion=model_direct_completion,
        )

        if seed_problems is None:
            logger.info("Calculating Seed Problems... w/o {}", completed)
            seed_problems = dataset_manager.find_seeds(
                k=seed_problems_k, metric=seed_problem_metric
            )

        for seed_problem in seed_problems:
            if seed_problem in (completed or []):
                logger.info(
                    f"Skipping problem {seed_problem} as it is already completed"
                )
                continue

            logger.info(f"Evaluating problem: {seed_problem}")
            try:
                Sampler.sample_problem_solutions(
                    inference_engine=inference_engine,
                    problem_id=seed_problem,
                    dataset_manager=dataset_manager,
                    canonical_samples=canonical_samples,
                    canonical_passing_threshold=canonical_passing_threshold,
                    scoring_samples=scoring_samples,
                    min_correct_samples=min_correct_samples,
                    exclude_mutation_types=exclude_mutation_types,
                    result_manager=result_manager,
                    base_only=base_only,
                    model_temps=model_temps,
                )
            except NoPassingSolutionException:
                logger.exception(
                    f"Unable to find passing solutions for problem {seed_problem}"
                )


class Evaluator:
    @staticmethod
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
            service_account_path: pathlib.Path = pathlib.Path(
                "/home/user/service-account.json"
            ),
    ):
        """
        Evaluate the completed stems generated by the model in GCS

        Args:
            dataset_name: which dataset to use
            model_name: model name to evaluate
            dataset_mini: whether to use mini version of the dataset
            dataset_noextreme: whether to exclude extreme samples from the dataset
            base_only: whether to evaluate base tests instead of plus dataset
            max_workers: number of workers to use for evaluating samples in parallel
            max_tasks: number of tasks before restarting each worker
            batch_size: number of samples to queue to the pool at a time
            restart_size: number of samples before restarting the pool
            gcs_bucket_name: gcs bucket name
            gcs_project_name: gcs project name
            service_account_path: GCS service account file path
        """
        logger.info("Evaluating Solutions...")
        result_manager = GCSResultStorageManager(
            model_name=model_name,
            bucket_name=gcs_bucket_name,
            project=gcs_project_name,
            service_account_file=str(service_account_path.absolute()),
        )
        dataset_manager = DatasetManager(
            dataset=dataset_name, mini=dataset_mini, noextreme=dataset_noextreme
        )
        for problem_id, eval_target, results in tqdm.tqdm(
                result_manager.get_data_pickles()
        ):
            evaluator = StemEvaluator(
                dataset_manager=dataset_manager,
                problem_id=problem_id,
                base_only=base_only,
                max_workers=max_workers,
                max_tasks=max_tasks,
                batch_size=batch_size,
                restart_size=restart_size,
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
        service_account_path: pathlib.Path = typer.Option(
            pathlib.Path("/home/user/service-account.json"),
            exists=True,
            help="Path to service account file.",
        ),
):
    Evaluator.evaluate_solutions(
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
        service_account_path=service_account_path,
    )


@app.command(name="sample")
def cli_sample_solutions(
        model_name: str = typer.Argument(..., help="The HF name of the model."),
        model_temps: str = typer.Option(
            ','.join(("0.3", "0.5", "0.7")), help="Temperatures to evaluate at."
        ),
        model_max_new_tokens: int = typer.Option(
            1024, help="Maximum number of new tokens the model can generate."
        ),
        tokenizer_name: str = typer.Option(
            None, help="The name of the tokenizer (defaults to model)"
        ),
        direct_completion: bool = typer.Option(
            False, help="Whether to use direct completion."
        ),
        # Codex used 0.95
        model_top_p: float = typer.Option(
            0.95, help="Top-p sampling parameter for the model.", min=0.0, max=1.0
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
            0.95, help="Passing threshold for canonical samples.", min=0.0, max=1.0
        ),
        canonical_samples: int = typer.Option(200, help="Number of canonical samples."),
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
        service_account_path: pathlib.Path = typer.Option(
            pathlib.Path("/home/user/service-account.json"),
            exists=True,
            help="Path to service account file.",
        ),
):
    Sampler.sample_solutions(
        model_name=model_name,
        model_temps=tuple(map(float, model_temps.split(','))),
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
        scoring_samples=pass_at_samples,
        min_correct_samples=canonical_min_correct_samples,
        seed_problems_k=seed_problems_k,
        seed_problem_metric=seed_problem_metric,
        seed_problems=seed_problems,
        exclude_mutation_types=exclude_mutation_types,
        gcs_bucket_name=gcs_bucket_name,
        gcs_project_name=gcs_project_name,
        completed=completed.split(","),
        service_account_path=service_account_path,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    app()
