import ast
from typing import Type

from mutations import CRT
from mutations.mutation import OneByOneTransformer
from mutations.visitor import OneByOneVisitor


class ReverseIterationVisitor(OneByOneVisitor):
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        new_call = ast.Call(
            func=ast.Name(id='range', ctx=ast.Load()),
            args=[
                ast.Call(
                    func=ast.Name(id='len', ctx=ast.Load()),
                    args=[node.args[0].args[0]],
                    keywords=[]
                ),
                ast.Constant(value=0),
                ast.Constant(value=-1)
            ],
            keywords=[]
        )
        return new_call

    def is_transformable(self, node):
        return (isinstance(node, ast.Call) and
                isinstance(node.func, ast.Name) and
                node.func.id == 'range' and
                len(node.args) == 1 and
                isinstance(node.args[0], ast.Call) and
                isinstance(node.args[0].func, ast.Name) and
                node.args[0].func.id == 'len' and
                len(node.args[0].args) == 1 and
                isinstance(node.args[0].args[0], ast.Name))


class ReverseIterationTransformer(OneByOneTransformer, category=CRT.arrays):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return ReverseIterationVisitor
