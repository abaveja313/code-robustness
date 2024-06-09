import ast
from typing import Callable
from asttokens import ASTTokens
from mutations import RegisteredTransformation, CRT
from shared.ast_utils import get_docstring_ranges, is_node_within_docstring


class StringQuoteTransformer:
    def __init__(self, source_code, old, new):
        self.old = old
        self.new = new
        self.source_code = source_code
        self.atok = ASTTokens(source_code, parse=True)
        self.tree = self.atok.tree
        self.docstring_lines = get_docstring_ranges(self.tree)

    def replace_quotes(self, target_node):
        original_text = self.atok.get_text(target_node)
        start, end = target_node.lineno, target_node.end_lineno
        col_start, col_end = target_node.col_offset, target_node.end_col_offset

        # Extract the exact string to replace
        before = self.source_code[:col_start]
        after = self.source_code[col_end:]
        target_string = self.source_code[col_start:col_end]

        # Replace the old quotes with the new quotes in the target string
        quoted_text = target_string.replace(self.old, self.new)

        # Construct the new source code with the specific replacement
        modified_source = before + quoted_text + after
        return modified_source

    def find_and_replace_strings(self):
        nodes = []

        def find_nodes(node):
            if (
                    isinstance(node, ast.Constant)
                    and isinstance(node.value, str)
                    and f"{self.old}" in self.atok.get_text(node)
                    and not is_node_within_docstring(node, self.docstring_lines)
            ):
                nodes.append(node)
            for child in ast.iter_child_nodes(node):
                find_nodes(child)

        find_nodes(self.tree)

        for node in nodes:
            yield self.replace_quotes(node)


class StringQuoteSingleTransformer(
    RegisteredTransformation, category=CRT.code_style, disabled=True
):
    @property
    def deterministic(self):
        return True

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        def process(source: str) -> list[str]:
            transformer = StringQuoteTransformer(source, old='"', new="'")
            return list(transformer.find_and_replace_strings())

        return process


class StringQuoteDoubleTransformer(
    RegisteredTransformation, category=CRT.code_style, disabled=False
):
    @property
    def deterministic(self):
        return True

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        def process(source: str) -> list[str]:
            transformer = StringQuoteTransformer(source, old="'", new='"')
            return list(transformer.find_and_replace_strings())

        return process
