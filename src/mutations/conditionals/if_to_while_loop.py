import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class IfToWhileLoopVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        while_loop = ast.While(test=node.test, body=[ast.Pass()], orelse=[])
        return while_loop

    def is_transformable(self, node):
        if not isinstance(node, ast.If):
            return False
        if not hasattr(node, 'parent'):
            return True
        return not (isinstance(node.parent, ast.If) and node in node.parent.orelse)


class IfToWhileLoopTransformer(OneByOneTransformer, category=CRT.conditionals):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IfToWhileLoopVisitor
