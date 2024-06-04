from abc import ABC, abstractmethod
from typing import Type, Callable

from loguru import logger

from inference.processors import Processors
from mutations.registry import RegisteredMixin
from mutations.visitor import OneByOneVisitor
from shared.structs import MutatedStem
from shared.program_utils import parse_stem


class RegisteredTransformation(RegisteredMixin, ABC, abstract=True):
    @property
    def deterministic(self):
        return True

    @property
    def stem_extra_skips(self):
        # Some mutations create extra lines we want to skip in our stem parsing.
        return 0

    @property
    @abstractmethod
    def attack_func(self) -> Callable[[str], list[str]]:
        pass

    def postprocess(self, original: str, mutated: list[str]) -> list[MutatedStem]:
        results = []
        for m in mutated:
            try:
                post_processed_original = Processors.postprocess_mutation(original)
                post_processed_mutated = Processors.postprocess_mutation(m)
            except Exception:
                logger.warning(
                    "Failed to postprocess sequence:\nOriginal:\n{}\nMutation:\n{}",
                    original,
                    m,
                )
                continue

            parsed = parse_stem(
                post_processed_original,
                post_processed_mutated,
                extra_skips=self.stem_extra_skips,
            )
            if not parsed:
                logger.warning("Skipping mutation as it had no effect")
                continue

            old_stem, new_stem = parsed
            stem = MutatedStem(original_stem=old_stem, mutated_stem=new_stem)
            results.append(stem)
        return results

    def get_transformations(self, current_text: str) -> list[MutatedStem]:
        # Filter if for some reason we have duplicate transformations
        # TODO do we still need this?
        transformed = list(set(self.attack_func(current_text)))
        post_processed: list[MutatedStem] = self.postprocess(current_text, transformed)
        logger.debug(
            f"{self.__class__.__name__} produced {len(post_processed)} transformations"
        )
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
