import ast
from abc import abstractmethod
from typing import Callable, List, Tuple

from asttokens import ASTTokens
from asttokens.util import replace

from mutations import CRT
from mutations import RegisteredTransformation
from shared.ast_utils import get_docstring_ranges, is_node_within_docstring


class IntegerConstantProcessor:
    def __init__(self, source_code: str, func: Callable[[int], str]):
        self.func = func
        self.source_code = source_code
        self.atok = ASTTokens(source_code, parse=True)
        self.tree = self.atok.tree
        self.docstring_ranges = get_docstring_ranges(self.tree)

    def replace_constant(self, target_node: ast.Constant) -> Tuple[int, int, str]:
        new_value = self.func(target_node.value)
        start, end = self.atok.get_text_range(target_node)
        return start, end, str(new_value)

    def find_and_replace_constants(self) -> List[str]:
        nodes = []

        def find_nodes(node: ast.AST):
            if (isinstance(node, ast.Constant) and isinstance(node.value, int) and
                    not isinstance(node.value, bool) and
                    not is_node_within_docstring(node, self.docstring_ranges)):
                nodes.append(node)
            for child in ast.iter_child_nodes(node):
                find_nodes(child)

        find_nodes(self.tree)

        modified_sources = []
        for node in nodes:
            replacement = [self.replace_constant(node)]
            modified_source = replace(self.source_code, replacement)
            modified_sources.append(modified_source)

        return modified_sources


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