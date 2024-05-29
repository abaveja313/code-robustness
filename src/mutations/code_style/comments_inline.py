from typing import Callable

import asttokens
from mutations import RegisteredTransformation, CRT

from shared.ast_utils import is_line_within_docstring, get_docstring_ranges


class InlineCommentsTransformer(RegisteredTransformation, category=CRT.code_style):

    @property
    def comment(self):
        return "I am a comment"

    @property
    def deterministic(self):
        return False

    def transform(self, code: str):
        results = []
        lines = code.split("\n")

        tree = asttokens.ASTTokens(code, parse=True).tree
        docstring_lines = get_docstring_ranges(tree)

        # Add inline comment to each indented line
        for i, line in enumerate(lines, start=1):
            if is_line_within_docstring(i, docstring_lines):
                continue
            if line.strip() == "" or not line.startswith(" "):
                continue  # Skip empty lines and non-indented lines
            copied = lines.copy()
            copied[i - 1] = line.rstrip() + f"  # {self.comment}"
            results.append("\n".join(copied))

        return results

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        return self.transform
