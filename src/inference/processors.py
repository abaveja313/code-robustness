import ast
import copy

import black

from shared.program_utils import (
    remove_pass,
    remove_comments_and_docstrings,
    fix_odd_indents,
    autopep8_normalize_ident,
    truncate_code_to_last_function,
)


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
        # Ensure that completion starts on a new line
        return remove_pass(stem).rstrip("\n") + "\n"

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
            return sequence
        except Exception as e:
            raise PostprocessingException(sequence) from e

    @staticmethod
    def postprocess_eval(sequence: str, direct: bool = False) -> str:
        original_sequence = copy.copy(sequence)

        if len(sequence.strip()) == 0:
            raise PostprocessingException(sequence) from Exception(
                "Refusing to postprocess empty code"
            )

        transforms = (
            lambda code: code.rstrip("\n").replace("\\\n", " "),
            # Only for direct completion we need to fix indentation because the model messes it up occasionally
            # ---
            lambda code: truncate_code_to_last_function(code) if direct else code,
            lambda code: fix_odd_indents(code) if direct else code,
            lambda code: autopep8_normalize_ident(code) if direct else code,
            # ---
            lambda code: remove_comments_and_docstrings(code, remove_docstrings=False),
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
            raise PostprocessingException(original_sequence) from e
        return sequence

    @staticmethod
    def split_sequences(
        sequences: list[str], sids: list[str], samples_per_sequence: int
    ):
        output = {}
        sid = 0
        for idx in range(0, len(sequences), samples_per_sequence):
            output[sids[sid]] = sequences[idx : idx + samples_per_sequence]
            sid += 1
        return output
