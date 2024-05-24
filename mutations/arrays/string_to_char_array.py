import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor


class StringToCharArrayVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        string_value = node.value.value
        char_nodes = [ast.Constant(value=char) for char in string_value]
        array_node = ast.List(elts=char_nodes, ctx=ast.Load())
        node.value = array_node
        return node

    def is_transformable(self, node):
        return (isinstance(node, ast.Assign) and
                isinstance(node.value, ast.Constant) and isinstance(node.value.value, str))


class StringToCharArrayTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return StringToCharArrayVisitor
