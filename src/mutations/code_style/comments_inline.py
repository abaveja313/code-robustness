from typing import Callable

from mutations import RegisteredTransformation, CRT


class LexicalCommentsInlineTransformer(RegisteredTransformation, category=CRT.code_style):
    @property
    def comment(self):
        return "I am a comment"

    @property
    def deterministic(self):
        return True

    def transform(self, code: str):
        results = []
        new_lines = code.split('\n')
        for idx in range(len(new_lines)):
            copied = new_lines.copy()
            # We only care about indented lines
            if not new_lines[idx].startswith('\t'):
                continue

            copied[idx] += f" # {self.comment}"
            results.append('\n'.join(copied))
        return results

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        return self.transform

