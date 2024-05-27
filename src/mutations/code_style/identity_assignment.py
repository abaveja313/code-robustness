import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT

from shared.ast_utils import is_unary_assign

from shared.ast_utils import StatementGroup


class IdentityAssignmentVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return is_unary_assign(node)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        identity_assign = ast.Assign(targets=[node.targets[0]], value=node.targets[0])
        return StatementGroup(body=[node, identity_assign])


class IdentityAssignmentTransformer(OneByOneTransformer, category=CRT.code_style):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IdentityAssignmentVisitor
