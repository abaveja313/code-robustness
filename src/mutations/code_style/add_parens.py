import ast
from typing import Callable

from asttokens import ASTTokens

from mutations import RegisteredTransformation, CRT

from shared.ast_utils import get_docstring_ranges, is_node_within_docstring


class ParensProcessor:
    def __init__(self, source_code, node_types=None):
        self.source_code = source_code
        self.node_types = (
            node_types
            if node_types
            else [ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare, ast.UnaryOp]
        )
        self.atok = ASTTokens(source_code, parse=True)
        self.tree = self.atok.tree
        self.docstring_ranges = get_docstring_ranges(self.tree)

    def add_redundant_parens(self, target_node):
        original_text = self.atok.get_text(target_node)
        paren_text = f"({original_text})"
        modified_source = self.source_code.replace(original_text, paren_text, 1)
        return modified_source

    def find_and_paren_nodes_recursive(self):
        nodes = []

        def find_nodes(node):
            if isinstance(
                node, tuple(self.node_types)
            ) and not is_node_within_docstring(node, self.docstring_ranges):
                nodes.append(node)
            for child in ast.iter_child_nodes(node):
                find_nodes(child)

        find_nodes(self.tree)

        for node in nodes:
            yield self.add_redundant_parens(node)


class AddParensTransformer(RegisteredTransformation, category=CRT.code_style):
    @property
    def deterministic(self):
        return True

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        def process(source) -> list[str]:
            parens_processor = ParensProcessor(source)
            return list(parens_processor.find_and_paren_nodes_recursive())

        return process
