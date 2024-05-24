import ast
from typing import Type

from mutations import (
    OneByOneTransformer,
    OneByOneVisitor, CRT,
)


class BooleanDemorgansVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        return self.apply_demorgans(node)

    @staticmethod
    def apply_demorgans(node):
        def apply_demorgans_expr(node):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not) and isinstance(node.operand, ast.BoolOp):
                if isinstance(node.operand.op, ast.And):
                    return ast.BoolOp(
                        op=ast.Or(),
                        values=[ast.UnaryOp(op=ast.Not(), operand=apply_demorgans_expr(value)) for value
                                in
                                node.operand.values])
                elif isinstance(node.operand.op, ast.Or):
                    return ast.BoolOp(op=ast.And(),
                                      values=[ast.UnaryOp(op=ast.Not(), operand=apply_demorgans_expr(value)) for value
                                              in
                                              node.operand.values])
            elif isinstance(node, ast.BoolOp):
                if all(isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not) for value in node.values):
                    if isinstance(node.op, ast.And):
                        return ast.BoolOp(op=ast.Or(),
                                          values=[apply_demorgans_expr(value.operand) for value in node.values])
                    elif isinstance(node.op, ast.Or):
                        return ast.BoolOp(op=ast.And(),
                                          values=[apply_demorgans_expr(value.operand) for value in node.values])
            return node

        if isinstance(node, ast.Expr):
            node.value = apply_demorgans_expr(node.value)
        elif isinstance(node, ast.If):
            node.test = apply_demorgans_expr(node.test)
        elif isinstance(node, ast.While):
            node.test = apply_demorgans_expr(node.test)
        elif isinstance(node, ast.Return):
            node.value = apply_demorgans_expr(node.value)
        elif isinstance(node, ast.Assign):
            node.targets = [apply_demorgans_expr(target) for target in node.targets]
            node.value = apply_demorgans_expr(node.value)
        return node

    @staticmethod
    def is_demorgans_simplifiable(node):
        def is_demorgans_simplifiable_expr(node):
            if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
                return isinstance(node.operand, ast.BoolOp) and isinstance(node.operand.op, (ast.And, ast.Or))
            elif isinstance(node, ast.BoolOp):
                return all(isinstance(value, ast.UnaryOp) and isinstance(value.op, ast.Not) for value in node.values)
            else:
                return False

        if isinstance(node, ast.Expr):
            return is_demorgans_simplifiable_expr(node.value)
        elif isinstance(node, ast.If):
            return is_demorgans_simplifiable_expr(node.test)
        elif isinstance(node, ast.While):
            return is_demorgans_simplifiable_expr(node.test)
        elif isinstance(node, ast.Return):
            return is_demorgans_simplifiable_expr(node.value)
        elif isinstance(node, ast.Assign):
            return any(
                is_demorgans_simplifiable_expr(target) for target in node.targets) or is_demorgans_simplifiable_expr(
                node.value)
        else:
            return False

    def is_transformable(self, node):
        return self.is_demorgans_simplifiable(node)


class BooleanDemorgansTransformer(OneByOneTransformer, category=CRT.booleans):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return BooleanDemorgansVisitor
