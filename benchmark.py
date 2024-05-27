from dataclasses import dataclass

import tqdm

from inference.models import make_model, VllmDecoder

from mutations import CRT, RegisteredMixin, RegisteredTransformation
from canonical import MaxProbInitializer
from shared.mutated_stem import MutatedStem
from inference.stem_evaluator import StemEvaluator
from loguru import logger


@dataclass
class BenchmarkResult:
    mutation: str
    original_prefix: str
    mutated_prefix: str
    pass_at_ratio: dict[str, float]

    def to_json(self):
        return {
            'mutation': self.mutation,
            'original_prefix': self.original_prefix,
            'mutated_prefix': self.mutated_prefix,
            'pass_at_ratio': self.pass_at_ratio
        }


def evaluate_problem(
        model: VllmDecoder,
        problem_id: str,
        dataset_params: dict[str, bool],
        canonical_samples: int,
        canonical_batch_size: int,
        scoring_samples: int,
        min_correct_samples: int,
        exclude_mutation_types: list[CRT] = None
):
    logger.info("Finding canonical solution...")
    initializer = MaxProbInitializer(
        model=model,
        problem_id=problem_id,
        num_samples=canonical_samples,
        batch_size=canonical_batch_size,
        min_correct_samples=min_correct_samples,
        **dataset_params
    )

    canonical_solution = initializer.canonical_solution()
    mutations: list[RegisteredTransformation] = RegisteredMixin.registry.get(
        exclude=exclude_mutation_types
    )
    evaluator = StemEvaluator(
        model=model,
        problem_id=problem_id,
        num_samples=scoring_samples,
        **dataset_params
    )

    results = []
    pbar = tqdm.tqdm(mutations)
    for mutation in pbar:
        pbar.set_description(f"{mutation.__class__.__name__}")
        stems: list[MutatedStem] = mutation.get_transformations(
            current_text=canonical_solution
        )
        for stem in tqdm.tqdm(stems):
            pass_ratios = evaluator.compute_log_pass_ratio(stem)
            result = BenchmarkResult(
                mutation=mutation.__class__.__name__,
                original_prefix=stem.original_stem,
                mutated_prefix=stem.mutated_stem,
                pass_at_ratio=pass_ratios
            )
            logger.info(result)
            results.append(result.to_json())

    return results


def benchmark(
        model_name: str,
        seed_problems: list[str],
        dataset_name: str,
        mini: bool = True,
        noextreme: bool = False,
        temperature: float = 0.5,
        canonical_samples: int = 200,
        canonical_batch_size: int = 50,
        scoring_samples: int = 100,
        min_correct_samples: int = 10,
        exclude_mutation_types: list[CRT] = None
):
    model = make_model(
        model_name,
        backend='vllm',
        temperature=temperature,
        dataset=dataset_name
    )
    dataset_params = {
        'mini': mini,
        'noextreme': noextreme
    }

    for seed_problem in seed_problems:
        logger.info("Evaluating problem: %s", seed_problem)
        evaluate_problem(
            model=model,
            problem_id=seed_problem,
            dataset_params=dataset_params,
            canonical_samples=canonical_samples,
            canonical_batch_size=canonical_batch_size,
            scoring_samples=scoring_samples,
            min_correct_samples=min_correct_samples,
            exclude_mutation_types=exclude_mutation_types
        )


if __name__ == "__main__":
    benchmark(
        model_name='deepseek-ai/deepseek-coder-1.3b-instruct',
        seed_problems=['Mbpp/771', 'Mbpp/100', 'Mbpp/129', 'Mbpp/245', 'Mbpp/306', 'Mbpp/239', 'Mbpp/123', 'Mbpp/20',
                       'Mbpp/721', 'Mbpp/71'],
        mini=False,
        noextreme=True
    )
