import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class DictToArrayVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        if isinstance(node, ast.Assign):
            if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                if isinstance(node.value, ast.Dict):
                    return len(node.value.keys) == 0
        return False

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Assign(targets=node.targets, value=ast.List(elts=[], ctx=ast.Load()))


class DictToArrayTransformer(OneByOneTransformer, category=CRT.dicts):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DictToArrayVisitor
