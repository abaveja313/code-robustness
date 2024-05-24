import ast
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor, CRT,
)


class ExpandBooleansVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.BoolOp(
            op=ast.And(),
            values=[node, ast.Constant(value=True)]
        )

    def is_transformable(self, node):
        return isinstance(node, ast.Compare) or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not)) or (
                isinstance(node, (ast.NameConstant, ast.Constant)) and isinstance(node.value, bool))


class ExpandBooleansTransformer(OneByOneTransformer, category=CRT.booleans):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ExpandBooleansVisitor
