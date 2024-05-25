import ast
from abc import abstractmethod, ABC
from typing import Type

from mutations import (
    OneByOneVisitor,
    OneByOneTransformer, CRT,
)


class MathInversionVisitor(OneByOneVisitor, ABC):
    @property
    @abstractmethod
    def op_type(self):
        pass

    def is_transformable(self, node):
        return isinstance(node, ast.BinOp) and isinstance(node.op, self.op_type)

    @abstractmethod
    def transform_node(self, node):
        pass


class AdditionInversionVisitor(MathInversionVisitor):
    @property
    def op_type(self):
        return ast.Add

    def transform_node(self, node):
        node.op = ast.Sub()
        node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
        return node


class SubtractionInversionVisitor(MathInversionVisitor):
    @property
    def op_type(self):
        return ast.Sub

    def transform_node(self, node):
        node.op = ast.Add()
        node.right = ast.UnaryOp(op=ast.USub(), operand=node.right)
        return node


class MultiplicationInversionVisitor(MathInversionVisitor):
    @property
    def op_type(self):
        return ast.Mult

    def transform_node(self, node):
        node.op = ast.Div()
        node.right = ast.BinOp(
            op=ast.Div(), left=ast.Constant(1.0), right=node.right
        )
        return node


class DivisionInversionVisitor(MathInversionVisitor):
    @property
    def op_type(self):
        return ast.Div

    def transform_node(self, node):
        node.op = ast.Mult()
        node.right = ast.BinOp(
            op=ast.Mult(), left=ast.Constant(1.0), right=node.right
        )
        return ast.Call(
            func=ast.Name(id="int", ctx=ast.Load()),
            args=[
                ast.BinOp(
                    left=node.left,
                    op=ast.Div(),
                    right=ast.BinOp(
                        left=ast.Constant(1.0), op=ast.Div(), right=node.right
                    ),
                )
            ],
            keywords=[],
        )


class ModuloInversionVisitor(MathInversionVisitor):
    @property
    def op_type(self):
        return ast.Mod

    def transform_node(self, node):
        node.op = ast.Sub()
        node.right = ast.BinOp(
            op=ast.Mult(),
            left=node.right,
            right=ast.BinOp(op=ast.FloorDiv(), left=node.left, right=node.right),
        )
        return node


class AdditionInversionTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return AdditionInversionVisitor


class SubtractionInversionTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return SubtractionInversionVisitor


class MultiplicationInversionTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return MultiplicationInversionVisitor


class DivisionInversionTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return DivisionInversionVisitor


class ModuloInversionTransformer(OneByOneTransformer, category=CRT.math):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ModuloInversionVisitor
