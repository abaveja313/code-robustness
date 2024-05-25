import ast
import copy
import string
import typing
from abc import ABC, abstractmethod
import random
from typing import Type

from mutations import OneByOneVisitor, OneByOneTransformer, CRT


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
            if isinstance(target, ast.Name):
                self.declared_idents.add(target.id)
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
        return (isinstance(node, ast.Assign) or isinstance(node, ast.FunctionDef)
                or isinstance(node, ast.For) or isinstance(node, ast.With))

    @abstractmethod
    def next_identifier(self) -> typing.Generator[str, None, None]:
        pass

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        next_ident = self.next_identifier()
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            node.targets[0].id = next(next_ident)
            return node
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name):
            node.target.id = next(next_ident)
            return node
        if isinstance(node, ast.With):
            mutated = []
            for new_node, new_item in one_by_one('items', node):
                if isinstance(new_item.optional_vars, ast.Name):
                    new_item.optional_vars.id = next(next_ident)
                    mutated.append(new_node)
            return mutated

        if isinstance(node, ast.FunctionDef):
            mutated = []
            for new_node, new_arg in one_by_one('args', node.args):
                new_arg.arg = next(next_ident)
                mutated.append(new_node)
            return mutated

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

    @property
    def name(self):
        return "LexicalIdentifierObfuscate"

    def next_identifier(self) -> typing.Generator[str, None, None]:
        while True:
            yield ''.join(random.choice(string.ascii_letters) for _ in range(self.var_size))


class IdentifierRenameTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IdentifierRenameVisitor


class IdentifierObfuscateTransformer(OneByOneTransformer, category=CRT.code_style):
    @property
    def visitor(self) -> Type[OneByOneVisitor]:
        return IdentifierObfuscateVisitor
