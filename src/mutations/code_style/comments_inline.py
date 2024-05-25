from typing import Callable
from mutations import RegisteredTransformation, CRT


class InlineCommentsTransformer(RegisteredTransformation, category=CRT.code_style):

    @property
    def comment(self):
        return "I am a comment"

    @property
    def deterministic(self):
        return False

    def transform(self, code: str):
        results = []
        lines = code.split('\n')

        # Add inline comment to each indented line
        for idx, line in enumerate(lines):
            if line.strip() == '' or not line.startswith(' '):
                continue  # Skip empty lines and non-indented lines
            copied = lines.copy()
            copied[idx] = line.rstrip() + f"  # {self.comment}"
            results.append('\n'.join(copied))

        return results

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        return self.transform

