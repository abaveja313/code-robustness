import textwrap
from enum import Enum

from evalplus.eval._special_oracle import MBPP_OUTPUT_NOT_NONE_TASKS
from evalplus.evaluate import (
    get_groundtruth,
    get_mbpp_plus,
    get_mbpp_plus_hash,
    get_human_eval_plus,
    get_human_eval_plus_hash,
)
from loguru import logger
from radon.complexity import cc_visit
from radon.metrics import h_visit
from radon.raw import analyze
from shared.ast_utils import get_function_declaration_line
from shared.program_utils import IDENT


class Dataset(str, Enum):
    HUMANEVAL = "humaneval"
    MBPP = "mbpp"


class SeedStrategy(str, Enum):
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    HALSTEAD_VOLUME = "halstead_volume"
    LOGICAL_LINES = "logical_lines"


class DatasetManager:
    def __init__(
            self, dataset: str = Dataset.MBPP, mini: bool = False, noextreme: bool = False
    ):
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
            MBPP_OUTPUT_NOT_NONE_TASKS if self.dataset_name == Dataset.MBPP else [],
        )

    def get_correct(self, problem_id):
        return self.ground_truth[problem_id]

    def get_problem(self, problem_id):
        return self.dataset[problem_id]

    def find_seeds(self, k: int, metric: str):
        match metric:
            case SeedStrategy.CYCLOMATIC_COMPLEXITY:
                m_func = lambda code: cc_visit(code)[0].complexity
            case SeedStrategy.HALSTEAD_VOLUME:
                m_func = lambda code: h_visit(code).total.volume
            case SeedStrategy.LOGICAL_LINES:
                m_func = lambda code: analyze(code).lloc
            case _:
                raise ValueError(f"Unknown metric: {metric}")

        solution_scores = {}
        for problem_id, detail in self.dataset.items():
            canonical_solution = detail["prompt"] + detail["canonical_solution"]
            complexity = m_func(canonical_solution)
            solution_scores[problem_id] = complexity

        seeds = sorted(solution_scores, key=solution_scores.get, reverse=True)[:k]

        logger.info(f"Dataset Seeds {self.dataset_name}: {seeds}")
        return seeds

    def format_prompts(self):
        # Format:
        # def function_name(arg1, arg2, ...):
        #   """
        #   prompt
        #   """
        if self.dataset_name == Dataset.HUMANEVAL:
            for problem_id in self.dataset:
                self.dataset[problem_id]["formatted_prompt"] = self.dataset[problem_id][
                    "prompt"
                ]
        elif self.dataset_name == Dataset.MBPP:
            for problem_id in self.dataset:
                prompt = self.dataset[problem_id]["prompt"]
                canonical = self.dataset[problem_id]["canonical_solution"]
                entry_point = self.dataset[problem_id]["entry_point"]
                function_declaration = get_function_declaration_line(
                    canonical, entry_point
                )
                indented_instructions = textwrap.indent(prompt, IDENT)
                formatted = f"{function_declaration}\n{indented_instructions}"
                self.dataset[problem_id]["formatted_prompt"] = formatted
