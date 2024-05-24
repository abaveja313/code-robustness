import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class StringToBytestringVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        # Encode the constant as a bytestring
        node.value.value = node.value.value.encode()
        return node

    def is_transformable(self, node):
        return (isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and
                isinstance(node.value.value, str))


class StringToByteStringTransformer(OneByOneTransformer, category=CRT.strings):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return StringToBytestringVisitor
