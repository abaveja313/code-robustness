import ast
import warnings
from typing import Type

from mutations import (
    OneByOneVisitor,
    OneByOneTransformer,
    CRT,
)


class EnumerateForVisitor(OneByOneVisitor):

    def is_transformable(self, node):
        return isinstance(node, ast.For) and not (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "enumerate"
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_iter = ast.Call(
            func=ast.Name(id="enumerate", ctx=ast.Load()), args=[node.iter], keywords=[]
        )

        # Add a first target loop variable called cidx
        if isinstance(node.target, ast.Name):
            new_target = ast.Tuple(
                elts=[ast.Name(id="cidx", ctx=ast.Store()), node.target],
                ctx=ast.Store(),
            )
        elif isinstance(node.target, ast.Tuple):
            new_target = ast.Tuple(
                elts=[ast.Name(id="cidx", ctx=ast.Store())] + node.target.elts,
                ctx=ast.Store(),
            )
        else:
            warnings.warn("Unsupported loop target!")
            # Do nothing!
            return node

        # Create a new For node with the updated iterable and target
        new_node = ast.For(
            target=new_target, iter=new_iter, body=node.body, orelse=node.orelse
        )

        return new_node


class EnumerateForTransformer(OneByOneTransformer, category=CRT.loops):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return EnumerateForVisitor
