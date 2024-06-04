import ast
from typing import Type

from mutations import (
    OneByOneVisitor,
    OneByOneTransformer,
    CRT,
)

from shared.ast_utils import StatementGroup


class ForToWhileVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return (
                isinstance(node, ast.For)
                and not isinstance(node.target, ast.Tuple)
                and isinstance(node.iter, ast.Call)
                and isinstance(node.iter.func, ast.Name)
                and node.iter.func.id == "range"
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        range_args = node.iter.args

        if len(range_args) == 1:  # range(stop)
            start = ast.Constant(0, lineno=node.lineno, col_offset=node.col_offset)
            stop = range_args[0]
            step = ast.Constant(1, lineno=node.lineno, col_offset=node.col_offset)
        elif len(range_args) == 2:  # range(start, stop)
            start = range_args[0]
            stop = range_args[1]
            step = ast.Constant(1, lineno=node.lineno, col_offset=node.col_offset)
        elif len(range_args) == 3:  # range(start, stop, step)
            start = range_args[0]
            stop = range_args[1]
            step = range_args[2]
        else:
            raise ValueError("Invalid number of arguments to `range`")

        index_var = node.target.id
        init_assign = ast.Assign(
            targets=[
                ast.Name(
                    id=index_var,
                    ctx=ast.Store(),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                )
            ],
            value=start,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        if isinstance(step, ast.Constant):
            step_value = step.value
            op = ast.Add() if step_value > 0 else ast.Sub()
            abs_step = ast.Constant(
                abs(step_value), lineno=node.lineno, col_offset=node.col_offset
            )
        elif isinstance(step, ast.UnaryOp) and isinstance(step.op, ast.USub):
            step_value = -step.operand.value
            op = ast.Sub()
            abs_step = ast.Constant(
                abs(step.operand.value), lineno=node.lineno, col_offset=node.col_offset
            )
        else:
            raise ValueError("Unsupported step type")

        condition = ast.Compare(
            left=ast.Name(
                id=index_var,
                ctx=ast.Load(),
                lineno=node.lineno,
                col_offset=node.col_offset,
            ),
            ops=[ast.Lt() if step_value > 0 else ast.Gt()],
            comparators=[stop],
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        while_body = node.body + [
            ast.AugAssign(
                target=ast.Name(
                    id=index_var,
                    ctx=ast.Store(),
                    lineno=node.lineno,
                    col_offset=node.col_offset,
                ),
                op=op,
                value=abs_step,
                lineno=node.lineno,
                col_offset=node.col_offset,
            )
        ]
        new_while = ast.While(
            test=condition,
            body=while_body,
            orelse=node.orelse,
            lineno=node.lineno,
            col_offset=node.col_offset,
        )

        return StatementGroup(body=[init_assign, new_while])


class ForToWhileTransformer(OneByOneTransformer, category=CRT.loops):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ForToWhileVisitor

    @property
    def stem_extra_skips(self):
        return 1
