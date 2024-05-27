import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class IntegerReplacementVisitor(OneByOneVisitor):
    @property
    def magic_constant(self):
        return 5

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_left = node.value + self.magic_constant
        return ast.BinOp(
            left=ast.Constant(value=new_left),
            op=ast.Add(),
            right=ast.Constant(value=-self.magic_constant)
        )

    def is_transformable(self, node):
        return isinstance(node, ast.Constant) and isinstance(node.value, int)


class IntegerReplacementTransformer(OneByOneTransformer, category=CRT.numbers):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IntegerReplacementVisitor
