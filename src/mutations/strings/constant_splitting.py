import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT
from shared.ast_utils import get_docstring_ranges, is_node_within_docstring


class ConstantSplittingVisitor(OneByOneVisitor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.docstring_ranges = get_docstring_ranges(self.ast_tree)

    @property
    def split_index(self):
        return 3

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        first_part = node.value[:3]
        rest_part = node.value[3:]

        # Create AST nodes for the parts of the string
        first_part_node = ast.Constant(value=first_part)
        rest_part_node = ast.Constant(value=rest_part)

        # Create a new AST node for the concatenated result
        concatenated_node = ast.BinOp(
            left=first_part_node, op=ast.Add(), right=rest_part_node
        )
        return concatenated_node

    def is_transformable(self, node):
        return (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and len(node.value) > 3
            and not is_node_within_docstring(node, self.docstring_ranges)
        )


class ConstantSplittingTransformer(OneByOneTransformer, category=CRT.strings):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ConstantSplittingVisitor
