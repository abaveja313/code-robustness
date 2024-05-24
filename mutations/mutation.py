from abc import ABC, abstractmethod
from typing import Type, Callable

from loguru import logger

from mutations.registry import RegisteredMixin
from mutations.visitor import OneByOneVisitor
from shared.mutated_stem import MutatedStem
from shared.program_utils import parse_stem


class RegisteredTransformation(RegisteredMixin, ABC, abstract=True):

    @property
    @abstractmethod
    def deterministic(self):
        pass

    @property
    @abstractmethod
    def attack_func(self) -> Callable[[str], list[str]]:
        pass

    def postprocess(self, original: str, mutated: list[str]) -> list[MutatedStem]:
        results = []
        for m in mutated:
            new_stem, old_stem = parse_stem(original, m)
            stem = MutatedStem(
                original_stem=old_stem,
                mutated_stem=m
            )
            results.append(stem)
        return results

    def get_transformations(self, current_text: str) -> list[MutatedStem]:
        transformed = self.attack_func(current_text)
        post_processed: list[MutatedStem] = self.postprocess(current_text, transformed)
        logger.debug(f"{self.__class__.__name__} produced {len(post_processed)} transformations")
        return post_processed


class OneByOneTransformer(RegisteredTransformation, ABC, abstract=True):
    @property
    def deterministic(self):
        return True

    @property
    @abstractmethod
    def visitor(self) -> Type[OneByOneVisitor]:
        pass

    def _visit_transform(self, source):
        return self.visitor(source).transform()

    @property
    def attack_func(self) -> Callable[[str], list[str]]:
        return self._visit_transform
