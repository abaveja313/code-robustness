import ast

import black

from shared.program_utils import remove_pass, remove_comments_and_docstrings


class PostprocessingException(Exception):
    def __init__(self, code: str, mutated: bool = False):
        self.code = code
        self.mutated = mutated
        super().__init__()


class Processors:
    @staticmethod
    def postprocess_canonical(code: str) -> str:
        return ast.unparse(ast.parse(code))

    @staticmethod
    def preprocess_stem(stem: str) -> str:
        return remove_pass(stem)

    @staticmethod
    def postprocess_mutation(sequence: str) -> str:
        transforms = (
            lambda code: code.rstrip("\n"),
            lambda code: black.format_str(
                code,
                mode=black.Mode(
                    string_normalization=False,
                    line_length=120,
                    magic_trailing_comma=False,
                ),
            ),
        )
        try:
            for transform in transforms:
                sequence = transform(sequence)

        except Exception as e:
            raise PostprocessingException(sequence) from e
        return sequence

    @staticmethod
    def postprocess_eval(sequence: str):
        transforms = (
            lambda code: code.rstrip("\n"),
            lambda c: remove_comments_and_docstrings(c, remove_docstrings=False),
            lambda code: black.format_str(
                code,
                mode=black.Mode(
                    string_normalization=False,
                    line_length=120,
                    magic_trailing_comma=False,
                ),
            ),
        )

        try:
            for transform in transforms:
                sequence = transform(sequence)

        except Exception as e:
            raise PostprocessingException(sequence) from e
        return sequence

    @staticmethod
    def split_sequences(sequences: list[str], sids: list[str], samples_per_sequence: int):
        output = {}
        sid = 0
        for idx in range(0, len(sequences), samples_per_sequence):
            output[sids[sid]] = sequences[idx: idx + samples_per_sequence]
            sid += 1
        return output
