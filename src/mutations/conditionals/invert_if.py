import ast
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor, CRT,
)


class InvertIfVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        condition = ast.UnaryOp(op=ast.Not(), operand=node.test)
        node.test = condition
        node.body = [ast.Pass()]
        return node

    def is_transformable(self, node):
        return isinstance(node, ast.If)


class InvertIfTransformer(OneByOneTransformer, category=CRT.conditionals):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return InvertIfVisitor
