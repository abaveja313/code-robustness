from shared.program_utils import remove_pass, remove_comments_and_docstrings, normalize_indentation
from loguru import logger


class PostprocessingException(Exception):
    def __init__(self, code: str, mutated: bool = False):
        self.code = code
        self.mutated = mutated
        super().__init__()


class Processors:
    @staticmethod
    def preprocess_stem(stem: str) -> str:
        return remove_pass(stem)

    @staticmethod
    def postprocess_sequence(sequence: str):
        transforms = (
            lambda c: remove_comments_and_docstrings(c, remove_docstrings=True),
            normalize_indentation,
        )
        exceptions = []
        for transform in transforms:
            sequence = transform(sequence)

        return sequence, exceptions

    @staticmethod
    def split_sequences(sequences: list[str], sids: list[str], samples_per_sequence: int):
        output = {}
        sid = 0
        for idx in range(0, len(sequences), samples_per_sequence):
            output[sids[sid]] = sequences[idx:idx + samples_per_sequence]
            sid += 1
        return output
