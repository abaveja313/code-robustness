import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor


class LenToGeneratorVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_call = ast.Call(
            func=ast.Name(id="sum", ctx=ast.Load()),
            args=[
                ast.ListComp(
                    elt=ast.Constant(value=1),
                    generators=[
                        ast.comprehension(
                            target=ast.Name(id="_", ctx=ast.Store()),
                            iter=node.args[0],
                            ifs=[],
                            is_async=0,
                        )
                    ],
                )
            ],
            keywords=[],
        )
        return new_call

    def is_transformable(self, node):
        return (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "len"
            and len(node.args) == 1
        )


class LenToGeneratorTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return LenToGeneratorVisitor
