from dataclasses import dataclass, field
from typing import Any


@dataclass
class MutatedStem:
    original_stem: str
    mutated_stem: str


@dataclass
class BenchmarkResult:
    problem_id: str
    mutation: str
    original_prefix: str = None
    mutated_prefix: str = None
    pass_at_original: dict[int, Any] = field(default_factory=dict)
    pass_at_mutated: dict[int, Any] = field(default_factory=dict)
    pass_at_ratio: dict[str, float] = field(default_factory=dict)
    passed_examples: list[str] = field(default_factory=list)
    failed_examples: list[str] = field(default_factory=list)
    bad_syntax_examples: list[str] = field(default_factory=list)

    def add_stem(self, stem: MutatedStem):
        self.original_prefix = stem.original_stem
        self.mutated_prefix = stem.mutated_stem

    def add_pass_ats(
        self, pass_at_original: dict[int, Any], pass_at_mutated: dict[int, Any]
    ):
        self.pass_at_original = pass_at_original
        self.pass_at_mutated = pass_at_mutated
