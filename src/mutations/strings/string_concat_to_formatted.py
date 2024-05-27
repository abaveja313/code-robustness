import ast
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


class StringConcatToFStringVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        components = self.collect_components(node)
        fstring_node = self.create_fstring(components)
        return fstring_node

    def collect_components(self, node):
        components = []
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            components.extend(self.collect_components(node.left))
            components.extend(self.collect_components(node.right))
        else:
            components.append(node)
        return components

    def create_fstring(self, components):
        formatted_values = []
        for component in components:
            if isinstance(component, ast.Constant) and isinstance(component.value, str):
                formatted_values.append(ast.Constant(value=component.value))
            else:
                formatted_values.append(
                    ast.FormattedValue(value=component, conversion=-1, format_spec=None)
                )

        return ast.JoinedStr(values=formatted_values)

    def is_transformable(self, node):
        return isinstance(node, ast.BinOp) and (
            (isinstance(node.left, ast.Constant) and isinstance(node.left.value, str))
            or (
                isinstance(node.right, ast.Constant)
                and isinstance(node.right.value, str)
            )
        )


class StringConcatToFStringTransformer(OneByOneTransformer, category=CRT.strings):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return StringConcatToFStringVisitor
