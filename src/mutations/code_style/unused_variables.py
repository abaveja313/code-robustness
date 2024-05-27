import ast
from random import random
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor,
    CRT,
)

from shared.ast_utils import StatementGroup


class UnusedVariableVisitor(OneByOneVisitor):

    @staticmethod
    def weird_assign():
        return ast.Assign(targets=[ast.Name("foo")], value=ast.Constant(value=3))

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return StatementGroup(body=[node, self.weird_assign()])

    def is_transformable(self, node):
        return (
            isinstance(node, (ast.Call, ast.Assign, ast.Expr))
            and hasattr(node, "parent")
            and isinstance(
                node.parent, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.Module)
            )
        )


class UnusedVariableTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return UnusedVariableVisitor
