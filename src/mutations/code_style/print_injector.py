import ast
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor,
    CRT,
)

from shared.ast_utils import StatementGroup


class PrintInjectionVisitor(OneByOneVisitor):
    @staticmethod
    def debug_statement():
        return ast.Expr(
            value=ast.Call(
                func=ast.Name(id="print", ctx=ast.Load()),
                args=[ast.Constant("This line was reached for debugging!")],
                keywords=[],
            )
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return StatementGroup(body=[node, self.debug_statement()])

    def is_transformable(self, node):
        return (
                isinstance(node, (ast.Call, ast.Assign, ast.Expr))
                and hasattr(node, "parent")
                and isinstance(
            node.parent, (ast.If, ast.For, ast.While, ast.FunctionDef, ast.Module)
        )
        )


class PrintInjectionTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return PrintInjectionVisitor

    @property
    def stem_extra_skips(self):
        # We are inserting an extra line
        return 1
