from dataclasses import dataclass, field
from typing import Any


@dataclass
class Solution:
    code: str
    probs: float
    failed_tests: list[int] = None

    def post_process(self):
        from inference.processors import Processors
        self.code = Processors.postprocess_eval(self.code)


class BatchSolution:
    def __init__(self, solutions: list[Solution] = None):
        self.solutions = solutions or []

    def get_code(self):
        return [solution.code for solution in self.solutions]

    def add(self, other: 'BatchSolution'):
        self.solutions.extend(other.solutions)


@dataclass
class MutatedStem:
    original_stem: str
    mutated_stem: str

    def as_tuple(self):
        return [('original', self.original_stem), ('mutated', self.mutated_stem)]


class SolutionType:
    PASSED = "passed"
    FAILED = "failed"
    BAD_SYNTAX = "bad_syntax"
    BAD_PROCESS = "bad_post_process"
    ERROR = "error"

    _ALL_TYPES = (PASSED, FAILED, BAD_SYNTAX, BAD_PROCESS, ERROR)


def create_examples():
    examples = {}
    for key in SolutionType._ALL_TYPES:
        examples.setdefault(key, {'original': [], 'mutated': []})
    return examples


@dataclass
class BenchmarkResult:
    problem_id: str
    mutation: str
    mutation_id: str
    stem_id: str
    original_prefix: str = None
    mutated_prefix: str = None
    pass_at_original: dict[int, Any] = field(default_factory=dict)
    pass_at_mutated: dict[int, Any] = field(default_factory=dict)
    pass_at_ratio: dict[str, float] = field(default_factory=dict)
    average_levenshtein: float = None
    examples: dict[str, dict[str, list[str]]] = field(default_factory=create_examples, repr=False)

    def add_stem(self, stem: MutatedStem):
        self.original_prefix = stem.original_stem
        self.mutated_prefix = stem.mutated_stem

    def add_pass_ats(
            self, pass_at_original: dict[int, Any], pass_at_mutated: dict[int, Any]
    ):
        self.pass_at_original = pass_at_original
        self.pass_at_mutated = pass_at_mutated

    def add_example(self, example, solution_type, mutated):
        sol_class = 'mutated' if mutated else 'original'
        self.examples[solution_type][sol_class].append(example)

    def compute_metrics(self):
        from shared.metrics import average_levenshtein_distance
        average_levenshtein = average_levenshtein_distance(self.examples['passed']['original'],
                                                           self.examples['passed']['mutated'])
        self.average_levenshtein = average_levenshtein
