import ast
from typing import Callable

from asttokens import ASTTokens

from mutations import RegisteredTransformation, CRT


class StringQuoteTransformer:
    def __init__(self, source_code, old, new):
        self.old = old
        self.new = new
        self.source_code = source_code
        self.atok = ASTTokens(source_code, parse=True)
        self.tree = self.atok.tree

    def replace_quotes(self, target_node):
        original_text = self.atok.get_text(target_node)
        quoted_text = original_text.replace(self.old, self.new)
        modified_source = self.source_code.replace(original_text, quoted_text, 1)
        return modified_source

    def find_and_replace_strings(self):
        nodes = []

        def find_nodes(node):
            if (
                isinstance(node, ast.Constant)
                and isinstance(node.value, str)
                and f"{self.old}" in self.atok.get_text(node)
            ):
                nodes.append(node)
            for child in ast.iter_child_nodes(node):
                find_nodes(child)

        find_nodes(self.tree)

        for node in nodes:
            yield self.replace_quotes(node)


class StringQuoteSingleTransformer(RegisteredTransformation, category=CRT.code_style):
    @property
    def deterministic(self):
        return True

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        def process(source: str) -> list[str]:
            transformer = StringQuoteTransformer(source, old='"', new="'")
            return list(transformer.find_and_replace_strings())

        return process


class StringQuoteDoubleTransformer(RegisteredTransformation, category=CRT.code_style):
    @property
    def deterministic(self):
        return True

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        def process(source: str) -> list[str]:
            transformer = StringQuoteTransformer(source, old="'", new='"')
            return list(transformer.find_and_replace_strings())

        return process
