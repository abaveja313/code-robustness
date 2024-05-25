import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class IdentityAssignmentVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and len(node.targets) == 1

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        identity_assign = ast.Assign(
            targets=[node.targets[0]],
            value=node.targets[0]
        )
        return ast.Module(body=[node, identity_assign])


class IdentityAssignmentTransformer(OneByOneTransformer, category=CRT.code_style):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IdentityAssignmentVisitor
