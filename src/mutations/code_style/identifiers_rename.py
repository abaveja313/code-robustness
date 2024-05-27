import ast
import copy
import string
import typing
from abc import ABC, abstractmethod
import random
from typing import Type

from mutations import CRT, OneByOneVisitor, OneByOneTransformer
from shared.ast_utils import is_unary_assign


def one_by_one(key: str, obj: object):
    if not hasattr(obj, key):
        raise ValueError(f"Object {obj} has no field {key}")

    for idx, value in enumerate(getattr(obj, key)):
        new_obj = copy.deepcopy(obj)
        yield new_obj, getattr(new_obj, key)[idx]


class DeclaredIdentifiersCollector(ast.NodeVisitor):
    def __init__(self):
        self.declared_idents = set()

    def visit_FunctionDef(self, node):
        for arg in node.args.args:
            self.declared_idents.add(arg.arg)
        self.generic_visit(node)

    def visit_Lambda(self, node):
        for arg in node.args.args:
            self.declared_idents.add(arg.arg)
        self.generic_visit(node)

    def visit_Assign(self, node):
        for target in node.targets:
            for node in ast.walk(target):
                if isinstance(target, ast.Name):
                    self.declared_idents.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            self.declared_idents.add(elt.id)
        self.generic_visit(node)

    def visit_For(self, node):
        if isinstance(node.target, ast.Name):
            self.declared_idents.add(node.target.id)
        self.generic_visit(node)

    def visit_With(self, node):
        for item in node.items:
            if isinstance(item.optional_vars, ast.Name):
                self.declared_idents.add(item.optional_vars.id)
        self.generic_visit(node)


def collect_declared_identifiers(source_code):
    tree = ast.parse(source_code)
    collector = DeclaredIdentifiersCollector()
    collector.visit(tree)
    return list(collector.declared_idents)


class IdentifierRenameVisitorBase(OneByOneVisitor, ABC):
    def is_transformable(self, node):
        return isinstance(node, (ast.FunctionDef, ast.For)) or is_unary_assign(node)

    @abstractmethod
    def next_identifier(self) -> typing.Generator[str, None, None]:
        pass

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        id_gen = self.next_identifier()
        if is_unary_assign(node):
            return ast.Assign(
                targets=[ast.Name(id=next(id_gen), ctx=ast.Store())], value=node.value
            )
        elif isinstance(node, ast.FunctionDef):
            for idx, arg in enumerate(node.args.args):
                arg.arg = next(id_gen)
            return node
        elif isinstance(node, ast.For):
            node.target.id = next(id_gen)
            return node
        return node


class IdentifierRenameVisitor(IdentifierRenameVisitorBase):
    def next_identifier(self) -> typing.Generator[str, None, None]:
        consumed_tokens = collect_declared_identifiers(self.source_code)
        existing_set = set(consumed_tokens)

        # Check single letter identifiers first
        for letter in string.ascii_lowercase:
            if letter not in existing_set:
                yield letter

        # Check double letter identifiers
        for first_letter in string.ascii_lowercase:
            for second_letter in string.ascii_lowercase:
                identifier = first_letter + second_letter
                if identifier not in existing_set:
                    yield identifier


class IdentifierObfuscateVisitor(IdentifierRenameVisitorBase):
    @property
    def var_size(self):
        return 8

    def next_identifier(self) -> typing.Generator[str, None, None]:
        while True:
            yield "".join(
                random.choice(string.ascii_letters) for _ in range(self.var_size)
            )


class IdentifierRenameTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IdentifierRenameVisitor


class IdentifierObfuscateTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IdentifierObfuscateVisitor
