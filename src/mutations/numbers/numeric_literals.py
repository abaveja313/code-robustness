import ast
import functools
from abc import abstractmethod
from typing import Callable, List

from asttokens import ASTTokens
from mutations import RegisteredTransformation

from mutations import CRT


class IntegerConstantProcessor:
    def __init__(self, source_code: str, func: Callable[[int], str]):
        self.func = func
        self.source_code = source_code
        self.atok = ASTTokens(source_code, parse=True)
        self.tree = self.atok.tree

    def replace_constant(self, target_node: ast.Constant, original_source: str) -> str:
        original_text = self.atok.get_text(target_node)
        new_value = self.func(target_node.value)
        modified_text = str(new_value)
        modified_source = original_source.replace(original_text, modified_text, 1)
        return modified_source

    def find_and_replace_constants(self) -> List[str]:
        nodes = []

        def find_nodes(node: ast.AST):
            if isinstance(node, ast.Constant) and isinstance(node.value, int):
                nodes.append(node)
            for child in ast.iter_child_nodes(node):
                find_nodes(child)

        find_nodes(self.tree)

        for node in nodes:
            modified_source = self.replace_constant(node, self.source_code)
            yield modified_source


class BaseIntegerConstantTransformer(RegisteredTransformation, abstract=True):
    @property
    @abstractmethod
    def func(self):
        pass

    def transform(self, source):
        transformer = IntegerConstantProcessor(source, self.func)
        return list(transformer.find_and_replace_constants())

    @property
    def attack_func(self) -> Callable[[str], List[str]]:
        return self.transform


class IntegerBinTransformer(BaseIntegerConstantTransformer, category=CRT.numbers):
    @property
    def func(self):
        return bin


class IntegerOctTransformer(BaseIntegerConstantTransformer, category=CRT.numbers):
    @property
    def func(self):
        return oct


class IntegerHexTransformer(BaseIntegerConstantTransformer, category=CRT.numbers):
    @property
    def func(self):
        return hex
