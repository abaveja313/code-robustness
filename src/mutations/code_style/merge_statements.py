import ast
from typing import Callable

from asttokens import asttokens

from mutations import (
    RegisteredTransformation,
    CRT,
)


class MergeConsecutiveStatements:
    def __init__(self, source):
        self.source = source
        self.atok = asttokens.ASTTokens(source, parse=True)
        self.tree = self.atok.tree

    def merge_statements(self, statements):
        merged_versions = []

        # Helper function to merge two statements
        def merge_two_statements(stmt1, stmt2):
            start1, end1 = self.atok.get_text_range(stmt1)
            start2, end2 = self.atok.get_text_range(stmt2)
            merged_code = self.source[start1:end1] + "; " + self.source[start2:end2]
            return self.source[:start1] + merged_code + self.source[end2:]

        i = 0
        while i < len(statements) - 1:
            stmt1, stmt2 = statements[i], statements[i + 1]
            if isinstance(stmt1, (ast.Expr, ast.Assign, ast.AugAssign)) and isinstance(
                stmt2, (ast.Expr, ast.Assign, ast.AugAssign)
            ):
                new_source = merge_two_statements(stmt1, stmt2)
                merged_versions.append(new_source)
            i += 1

        return merged_versions

    def process_node(self, node):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.If, ast.For, ast.While)):
            # Process main body
            merged_bodies = self.merge_statements(node.body)
            for merged_body in merged_bodies:
                yield merged_body

            # Process orelse body if exists
            if hasattr(node, "orelse") and node.orelse:
                merged_orelse = self.merge_statements(node.orelse)
                for merged_body in merged_orelse:
                    yield merged_body

        for child in ast.iter_child_nodes(node):
            yield from self.process_node(child)

    def get_merged_versions(self):
        unique_versions = set()
        for node in ast.walk(self.tree):
            for version in self.process_node(node):
                if version not in unique_versions:
                    unique_versions.add(version)
                    yield version


class MergeStatementsTransformer(RegisteredTransformation, category=CRT.code_style):

    @property
    def deterministic(self):
        return True

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        def process(source: str) -> list[str]:
            merger = MergeConsecutiveStatements(source=source)
            return list(merger.get_merged_versions())

        return process
