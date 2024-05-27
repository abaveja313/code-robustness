import textwrap

from mutations import RegisteredTransformation, CRT


class BlockCommentsTransformer(RegisteredTransformation, category=CRT.code_style):

    @property
    def comment(self):
        return "# I am a comment\n# I am a comment"

    @property
    def deterministic(self):
        return True  # Set to False to generate different variations

    def transform(self, code: str):
        new_lines = code.split("\n")
        results = []

        for i, line in enumerate(new_lines):
            if len(line.strip()) == 0:
                continue  # Skip empty lines
            indentation_level = len(line) - len(line.lstrip())
            if indentation_level > 0:
                copied = new_lines.copy()
                indented_comment = [
                    " " * indentation_level + cmt for cmt in self.comment.split("\n")
                ]
                copied.insert(i, "\n".join(indented_comment))
                results.append("\n".join(copied))

        return results

    @property
    def attack_func(self) -> callable:
        return self.transform
