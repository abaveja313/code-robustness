import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor


class NestedArrayInitializerVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and isinstance(node.value, ast.List)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.List(
            elts=[node.value],
            ctx=ast.Load()
        )


class NestedArrayInitializerTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return NestedArrayInitializerVisitor
