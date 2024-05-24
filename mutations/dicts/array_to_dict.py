import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class ArrayToDictVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and isinstance(node.value, ast.List) and len(node.value.elts) == 0

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Assign(
            targets=node.targets,
            value=ast.Dict(
                keys=[],
                values=[],
                ctx=ast.Load()
            )
        )


class ArrayToDictTransformer(OneByOneTransformer, category=CRT.dicts):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ArrayToDictVisitor
