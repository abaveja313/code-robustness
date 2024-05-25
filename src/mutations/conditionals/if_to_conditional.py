import ast
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor, CRT,
)


class IfToConditionalVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        if_body = node.body[0]
        else_body = node.orelse[0]

        target = if_body.targets[0]
        if_value = if_body.value
        else_value = else_body.value

        ifexpr = ast.IfExp(
            test=node.test,
            body=if_value,
            orelse=else_value
        )

        assign = ast.Assign(
            targets=[target],
            value=ifexpr
        )

        return assign

    def is_transformable(self, node):
        if isinstance(node, ast.If):
            if len(node.body) == 1 and len(node.orelse) == 1:
                if_body = node.body[0]
                else_body = node.orelse[0]

                if isinstance(if_body, ast.Assign) and isinstance(else_body, ast.Assign):
                    if len(if_body.targets) == 1 and len(else_body.targets) == 1:
                        if_target = if_body.targets[0]
                        else_target = else_body.targets[0]

                        if isinstance(if_target, ast.Name) and isinstance(else_target, ast.Name):
                            return if_target.id == else_target.id

        return False


class IfToConditionalTransformer(OneByOneTransformer, category=CRT.conditionals):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IfToConditionalVisitor
