import ast
from typing import Type

from mutations import (
    OneByOneVisitor,
    OneByOneTransformer, CRT,
)


class StringConcatToJoinVisitor(OneByOneVisitor):

    @property
    def name(self):
        return "StringConcatToJoin"

    def is_transformable(self, node):
        return (
            isinstance(node, ast.BinOp)
            and isinstance(node.op, ast.Add)
            and (isinstance(node.left, ast.Str) or isinstance(node.right, ast.Str))
        )

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        elements = []

        def collect_concat_parts(n):
            if isinstance(n, ast.BinOp) and isinstance(n.op, ast.Add):
                collect_concat_parts(n.left)
                collect_concat_parts(n.right)
            else:
                # Convert non-string types to string using str()
                if not isinstance(n, ast.Str):
                    n = ast.Call(
                        func=ast.Name(id="str", ctx=ast.Load()), args=[n], keywords=[]
                    )
                elements.append(n)

        collect_concat_parts(node)

        # Create the new ast.Call node for 'join'
        join_call = ast.Call(
            func=ast.Attribute(value=ast.Str(s=""), attr="join", ctx=ast.Load()),
            args=[ast.List(elts=elements, ctx=ast.Load())],
            keywords=[],
        )

        return join_call


class StringConcatToJoinTransformer(OneByOneTransformer, category=CRT.strings):

    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return StringConcatToJoinVisitor
