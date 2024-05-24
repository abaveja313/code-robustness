import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class DictToArrayVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return (isinstance(node, ast.Assign) and isinstance(node.value, ast.Dict) and len(node.value.keys) == 0 and len(
            node.value.values) == 0)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Assign(
            targets=node.targets,
            value=ast.List(
                elts=[],
                ctx=ast.Load()
            )
        )


class DictToArrayTransformer(OneByOneTransformer, category=CRT.dicts):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DictToArrayVisitor
