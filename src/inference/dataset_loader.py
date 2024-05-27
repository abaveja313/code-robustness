import textwrap
from dataclasses import dataclass
from enum import Enum, auto
from evalplus.evaluate import get_groundtruth, get_mbpp_plus, get_mbpp_plus_hash, get_human_eval_plus, \
    get_human_eval_plus_hash
from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from loguru import logger
from radon.metrics import mi_parameters
from shared.ast_utils import get_function_declaration_line


class Dataset:
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


class DatasetManager:
    def __init__(self, dataset: str = Dataset.MBPP, mini: bool = False, noextreme: bool = False):
        self.dataset_name = dataset
        self.dataset_params = dict(mini=mini, noextreme=noextreme)

        self.dataset = None
        self.dataset_hash = None
        self.ground_truth = None

        self.load_dataset()
        self.load_groundtruth()
        self.format_prompts()

    def load_dataset(self):
        if self.dataset_name == Dataset.HUMANEVAL:
            self.dataset = get_human_eval_plus(**self.dataset_params)
            self.dataset_hash = get_human_eval_plus_hash(**self.dataset_params)
        elif self.dataset_name == Dataset.MBPP:
            self.dataset = get_mbpp_plus(**self.dataset_params)
            self.dataset_hash = get_mbpp_plus_hash(**self.dataset_params)
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

    def load_groundtruth(self):
        self.ground_truth = get_groundtruth(
            self.dataset,
            self.dataset_hash,
            MBPP_OUTPUT_NOT_NONE_TASKS if self.dataset_name == Dataset.MBPP else []
        )

    def get_correct(self, problem_id):
        return self.ground_truth[problem_id]

    def get_problem(self, problem_id):
        return self.dataset[problem_id]

    def find_seeds(self, k: int, metric: str):
        metric_mapping = {
            'cyclomatic_complexity': 0,
            'halstead_volume': 1,
            'logical_lines': 2
        }
        assert metric in metric_mapping, "Metric not supported"

        solution_scores = {}
        for problem_id, detail in self.dataset.items():
            canonical_solution = detail['prompt'] + detail['canonical_solution']
            complexity = mi_parameters(canonical_solution)
            solution_scores[problem_id] = complexity[metric_mapping[metric]]

        seeds = sorted(solution_scores, key=solution_scores.get, reverse=True)[:k]
        logger.info(f"Dataset Seeds {self.dataset_name}: {seeds}")
        return seeds

    def format_prompts(self):
        if self.dataset_name == Dataset.HUMANEVAL:
            return

        for problem_id in self.dataset:
            prompt = self.dataset[problem_id]['prompt']
            canonical = self.dataset[problem_id]['canonical_solution']
            entry_point = self.dataset[problem_id]['entry_point']
            function_declaration = get_function_declaration_line(
                canonical, entry_point
            )
            indented_instructions = textwrap.indent(prompt, ' ' * 4)
            formatted = f"{function_declaration}\n{indented_instructions}"
            logger.info(f"Replacing Prompt for Problem ID: {problem_id}.")
            self.dataset[problem_id]['prompt'] = formatted
