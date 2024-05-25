import ast
from typing import Type

from mutations import (
    OneByOneVisitor,
    OneByOneTransformer, CRT,
)


class WhileToIfVisitor(OneByOneVisitor):

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.If(
            body=[ast.Pass()],
            test=node.test
        )

    def is_transformable(self, node):
        return isinstance(node, ast.While)


class WhileToIfTransformer(OneByOneTransformer, category=CRT.loops):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return WhileToIfVisitor
