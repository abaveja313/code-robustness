from asttokens import asttokens
from mutations import RegisteredTransformation, CRT
from shared.ast_utils import get_docstring_ranges, is_line_within_docstring


class BlockCommentsTransformer(RegisteredTransformation, category=CRT.code_style):

    @property
    def comment(self):
        return "# I am a block comment\n# I am a block comment"

    @property
    def deterministic(self):
        return True  # Set to False to generate different variations

    def transform(self, code: str):
        new_lines = code.split("\n")
        results = []
        tree = asttokens.ASTTokens(code, parse=True).tree
        docstring_lines = get_docstring_ranges(tree)

        for i, line in enumerate(new_lines, start=1):
            if is_line_within_docstring(i, docstring_lines):
                continue
            if len(line.strip()) == 0:
                continue  # Skip empty lines
            indentation_level = len(line) - len(line.lstrip())
            if indentation_level > 0:
                copied = new_lines.copy()
                indented_comment = [
                    " " * indentation_level + cmt for cmt in self.comment.split("\n")
                ]
                copied.insert(i - 1, "\n".join(indented_comment))
                results.append("\n".join(copied))

        return results

    @property
    def attack_func(self) -> callable:
        return self.transform
