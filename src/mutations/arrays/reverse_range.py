import ast
from typing import Type, Tuple

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor

from shared.ast_utils import get_constant_num


class ReverseRangeVisitor(OneByOneVisitor):
    def extract_range_params(
        self, node
    ) -> Tuple[ast.Constant, ast.Constant, ast.Constant]:
        args = node.args
        num_args = len(args)

        if num_args == 1:
            # range(stop)
            return ast.Constant(value=0), args[0], ast.Constant(value=1)
        elif num_args == 2:
            # range(start, stop)
            return args[0], args[1], ast.Constant(value=1)
        elif num_args == 3:
            # range(start, stop, step)
            return args[0], args[1], args[2]
        else:
            raise ValueError("range function call must have 1 to 3 arguments")

    def reverse_range_params(self, start, stop, step):
        # We use this for non-constant start/stop values by applying a BinOp to the node
        # ex. range(a, b) -> range(b-1, a-1, -1)
        op = ast.Sub() if step.value > 0 else ast.Add()

        # Use this for constant start/stop values by calculating the new value directly
        add_factor = -1 if step.value > 0 else 1

        # If the start is a constant, we can calculate the new start value
        if isinstance(start, ast.Constant):
            new_stop = ast.Constant(value=start.value + add_factor)
        else:
            new_stop = ast.BinOp(left=start, op=op, right=ast.Constant(value=1))

        if isinstance(stop, ast.Constant):
            new_start = ast.Constant(value=stop.value + add_factor)
        else:
            new_start = ast.BinOp(left=stop, op=op, right=ast.Constant(value=1))

        return new_start, new_stop, ast.Constant(value=-1 * step.value)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        # We need to apply get_constant_num to map UnaryOp(USub) to a negative constant
        start, stop, step = tuple(
            map(get_constant_num, self.extract_range_params(node))
        )
        args = self.reverse_range_params(start, stop, step)

        new_call = ast.Call(
            func=ast.Name(id="range", ctx=ast.Load()),
            args=args,
            keywords=[],
        )
        return new_call

    def has_constant_step(self, node):
        # If the step is dynamic, we have no way of knowing whether it is positive or negative
        # and thus we cannot reverse the range
        return 1 <= len(node.args) < 3 or (
            len(node.args) == 3
            and isinstance(get_constant_num(node.args[2]), ast.Constant)
        )

    def is_transformable(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "range"
            and self.has_constant_step(node)
        )


class ReverseIterationTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ReverseRangeVisitor
