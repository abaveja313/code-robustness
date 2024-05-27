import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor
from shared.ast_utils import is_unary_assign


class NestedArrayInitializerVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return is_unary_assign(node) and isinstance(node.value, ast.List)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Assign(
            targets=node.targets, value=ast.List(elts=[node.value], ctx=ast.Load())
        )


class NestedArrayInitializerTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return NestedArrayInitializerVisitor
