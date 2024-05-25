import ast
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor, CRT,
)


class PrintInjectionVisitor(OneByOneVisitor):

    @staticmethod
    def debug_statement():
        return ast.Call(
            func=ast.Name(id="print", ctx=ast.Load()),
            args=[ast.Constant("This line was reached for debugging!")],
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return ast.Module(body=[node, self.debug_statement()])

    def is_transformable(self, node):
        return isinstance(node, ast.Expr) or isinstance(node, ast.Assign)


class PrintInjectionTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return PrintInjectionVisitor
