from pprint import pprint

import numpy as np
from evalplus.data import get_human_eval_plus, get_mbpp_plus
from radon.metrics import mi_parameters


def compute_source_metrics(dataset: dict[str, dict]):
    solution_scores = {}
    for problem_id, detail in dataset.items():
        canonical_solution = detail['prompt'] + detail['canonical_solution']
        complexity = mi_parameters(canonical_solution)
        solution_scores[problem_id] = {
            'halstead_volume': complexity[1],
            'cyclomatic_complexity': complexity[0]
        }
    return solution_scores


def summarize(mapping: dict[str, float]):
    values = list(mapping.values())
    columns = {
        'argmax': max(mapping, key=mapping.get),
        'max': max(mapping.values()),
        'min': min(mapping.values()),
        'argmin': min(mapping, key=mapping.get),
        'mean': np.mean(values),
        'std': np.std(values),
        'p25': np.percentile(values, 25),
        'p75': np.percentile(values, 75),
        'p90': np.percentile(values, 90),
    }

    return columns


def main():
    k = 10
    mb_dataset = get_mbpp_plus()
    mb_dataset_metrics = compute_source_metrics(mb_dataset)
    m = {key: mb_dataset_metrics[key]['halstead_volume'] for key in mb_dataset_metrics}
    pprint(summarize(m))
    print(sorted(m, key=m.get, reverse=True)[:k])

    k = 4
    he_dataset = get_human_eval_plus()
    he_dataset_metrics = compute_source_metrics(he_dataset)
    m = {key: he_dataset_metrics[key]['halstead_volume'] for key in he_dataset_metrics}
    pprint(summarize(m))
    print(sorted(m, key=m.get, reverse=True)[:k])


if __name__ == "__main__":
    main()
