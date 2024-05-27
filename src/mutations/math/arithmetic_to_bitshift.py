import ast
from typing import Type

from mutations import OneByOneTransformer, OneByOneVisitor, CRT


class MultiplyBy2ToBitshiftVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Mult)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 2
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        node.op = ast.LShift()
        node.right = ast.Constant(value=1)
        return node


class DivideBy2ToBitshiftVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Div)
            and isinstance(node.right, ast.Constant)
            and node.right.value == 2
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        node.op = ast.RShift()
        node.right = ast.Constant(value=1)
        return node


class NegationToComplementVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_node = ast.BinOp(
            op=ast.Add(),
            left=ast.UnaryOp(op=ast.Invert(), operand=node.operand),
            right=ast.Constant(value=1),
        )
        return new_node


class MultiplyBy2ToBitshiftTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return MultiplyBy2ToBitshiftVisitor


class DivideBy2ToBitshiftTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DivideBy2ToBitshiftVisitor


class NegationToComplementTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return NegationToComplementVisitor
