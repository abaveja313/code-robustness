import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT

from shared.ast_utils import is_unary_assign


class DictInitializerUnpackVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        empty_dict = ast.Dict(keys=[], values=[])

        merged_dict = ast.Dict(
            keys=[None, None],
            values=[
                node.value,
                empty_dict
            ]
        )

        new_node = ast.Assign(targets=node.targets, value=merged_dict)
        return new_node

    def is_transformable(self, node):
        return is_unary_assign(node) and isinstance(node.value, ast.Dict)


class DictInitializerUnpackTransformer(OneByOneTransformer, category=CRT.dicts):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DictInitializerUnpackVisitor
