import ast
import copy
from typing import Type

from loguru import logger

from mutations import OneByOneVisitor, OneByOneTransformer, CRT
from shared.ast_utils import is_unary_assign


class SimplifyTransformer(ast.NodeTransformer):
    def visit_UnaryOp(self, node):
        node.operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(
                    node.operand.op, ast.Not
            ):
                # Rule 1: A unary not applied to a unary not can be replaced with its value
                new_node = node.operand.operand
                return self.generic_visit(new_node)
            elif (
                    isinstance(node.operand, ast.NameConstant)
                    and node.operand.value is False
            ):
                # Rule: not False can be replaced with True
                new_node = ast.NameConstant(value=True)
                return self.generic_visit(new_node)
            elif isinstance(node.operand, ast.Compare):
                # Rule 2: A not applied to any one of these expressions can be replaced with its opposite operation
                new_ops = []
                for op in node.operand.ops:
                    if isinstance(op, ast.Gt):
                        new_ops.append(ast.LtE())
                    elif isinstance(op, ast.Lt):
                        new_ops.append(ast.GtE())
                    elif isinstance(op, ast.GtE):
                        new_ops.append(ast.Lt())
                    elif isinstance(op, ast.LtE):
                        new_ops.append(ast.Gt())
                    elif isinstance(op, ast.Is):
                        new_ops.append(ast.IsNot())
                    elif isinstance(op, ast.IsNot):
                        new_ops.append(ast.Is())
                    elif isinstance(op, ast.In):
                        new_ops.append(ast.NotIn())
                    elif isinstance(op, ast.NotIn):
                        new_ops.append(ast.In())
                    elif isinstance(op, ast.Eq):
                        new_ops.append(ast.NotEq())
                    elif isinstance(op, ast.NotEq):
                        new_ops.append(ast.Eq())
                new_node = ast.Compare(
                    left=node.operand.left,
                    ops=new_ops,
                    comparators=node.operand.comparators,
                )
                return self.generic_visit(new_node)
            elif isinstance(node.operand, ast.BoolOp):
                # Rule 3: Apply De Morgan's law if we find a not applied to a BoolOp
                new_op = ast.And() if isinstance(node.operand.op, ast.Or) else ast.Or()
                new_values = [
                    ast.UnaryOp(op=ast.Not(), operand=self.visit(value))
                    for value in node.operand.values
                ]
                new_node = ast.BoolOp(op=new_op, values=new_values)
                return self.generic_visit(new_node)
        return node

    def visit_BoolOp(self, node):
        node.values = [self.visit(value) for value in node.values]
        return node

    def visit_Compare(self, node):
        node.left = self.visit(node.left)
        node.comparators = [self.visit(comparator) for comparator in node.comparators]
        return node


class FirstInversionVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, (ast.If, ast.While))

    def transform_node(self, node: ast.AST):
        if isinstance(node, (ast.If, ast.While)):
            node.test = self.transform_expression(node.test)
        return [node]

    @staticmethod
    def transform_expression(node: ast.expr):
        # Step 1: If the expression is a unary not on a compound expression
        # (bool op), apply De Morgan's law to expand it
        if (
                isinstance(node, ast.UnaryOp)
                and isinstance(node.op, ast.Not)
                and isinstance(node.operand, ast.BoolOp)
        ):
            node = FirstInversionVisitor.apply_demorgan(node.operand)

        # Step 3: Place a unary not around the entire expression from step 2
        node = ast.UnaryOp(op=ast.Not(), operand=node)

        # Step 6: If you have a bool op, apply step 7 to each of the terms
        # of the bool op. Otherwise, apply it to the output from step 5
        if isinstance(node, ast.BoolOp):
            node.values = [
                FirstInversionVisitor.simplify_expression(value)
                for value in node.values
            ]
        else:
            node = FirstInversionVisitor.simplify_expression(node)

        return node

    @staticmethod
    def apply_demorgan(node: ast.BoolOp):
        if isinstance(node.op, ast.And):
            new_op = ast.Or()
        else:
            new_op = ast.And()
        new_values = [ast.UnaryOp(op=ast.Not(), operand=value) for value in node.values]
        return ast.BoolOp(op=new_op, values=new_values)

    @staticmethod
    def simplify_expression(node: ast.expr):
        simplify_transformer = SimplifyTransformer()
        node = simplify_transformer.visit(node)
        return node


class SecondInversionVisitor(OneByOneVisitor):
    def is_boolean_expr(self, node):
        return (
                isinstance(node, (ast.Constant, ast.NameConstant))
                and isinstance(getattr(node, "value", None), bool)
                or isinstance(node, (ast.Compare, ast.BoolOp))
                or (isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not))
                or (
                        isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id in {"isinstance", "callable", "bool"}
                )
        )

    def is_transformable(self, node):
        return (
                isinstance(node, (ast.If, ast.While))
                or (is_unary_assign(node) and self.is_boolean_expr(node.value))
                or (isinstance(node, ast.Assign) and self.is_boolean_expr(node.value))
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        unmodified = copy.deepcopy(node)
        if isinstance(node, (ast.If, ast.While)):
            node.test = FirstInversionVisitor.transform_expression(node.test)
            node.test = ast.UnaryOp(op=ast.Not(), operand=node.test)
        elif isinstance(node, (ast.Assign, ast.Return)):
            node.value = FirstInversionVisitor.transform_expression(node.value)
            node.value = ast.UnaryOp(op=ast.Not(), operand=node.value)

        # This occurs when we double a simple boolean expression that is already negated (we get back the original)
        if ast.dump(unmodified) == ast.dump(node):
            logger.warning(f"Mutation on {ast.dump(node)} did not change the AST")
            return []

        return [node]


class FirstInversionTransformer(OneByOneTransformer, category=CRT.booleans):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return FirstInversionVisitor


class SecondInversionTransformer(OneByOneTransformer, category=CRT.booleans):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return SecondInversionVisitor
