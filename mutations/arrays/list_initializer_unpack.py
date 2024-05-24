import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor


class ListInitializerUnpackVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_value = ast.Call(
            func=ast.Name(id='list', ctx=ast.Load()),
            args=[ast.Starred(value=ast.List(elts=[node.value], ctx=ast.Load()), ctx=ast.Load())],
            keywords=[]
        )
        new_assign = ast.Assign(
            targets=node.targets,
            value=new_value,
            lineno=node.lineno,
            col_offset=node.col_offset
        )
        return new_assign

    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and isinstance(node.value, ast.List)


class ListInitializerUnpackTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ListInitializerUnpackVisitor
