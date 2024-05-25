import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class OverflowIntegerVisitor(OneByOneVisitor):
    @property
    def magic_constant(self):
        return 1e20

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_node = ast.BinOp(
            left=ast.BinOp(
                left=node,
                op=ast.Mult(),
                right=ast.Constant(value=self.magic_constant)
            ),
            op=ast.FloorDiv(),
            right=ast.Constant(value=self.magic_constant)
        )
        node.value = new_node
        return node

    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value,
                                                                                                    int)


class OverflowIntegerTransformer(OneByOneTransformer, category=CRT.numbers):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return OverflowIntegerVisitor
