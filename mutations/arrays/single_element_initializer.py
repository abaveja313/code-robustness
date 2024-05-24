import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor


class SingleElementInitializerVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and len(node.value.elts) == 0

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Assign(
            targets=node.targets,
            value=ast.List(
                elts=[ast.Constant(value=None)],
                ctx=ast.Load()
            )
        )


class SingleElementInitializerTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return SingleElementInitializerVisitor
