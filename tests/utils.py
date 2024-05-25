import textwrap
from typing import Type

from mutations import OneByOneVisitor, RegisteredTransformation
from shared.program_utils import normalize_indentation


def normalize(src):
    src = normalize_indentation(src)
    return src


def verify(results, expected):
    if not isinstance(expected, list):
        expected = [expected]

    normalized_results = set(map(normalize, results))
    normalized_expected = set(map(normalize, expected))

    # print("Results:")
    # for i, result in enumerate(results, start=1):
    #     print(f"Result {i}:")
    #     print(result)
    #     print()
    #
    # print("Expected:")
    # for i, exp in enumerate(expected, start=1):
    #     print(f"Expected {i}:")
    #     print(exp)
    #     print()

    missing_results = normalized_expected - normalized_results
    extra_results = normalized_results - normalized_expected

    assert normalized_results == normalized_expected, (
            "Results do not match the expected values.\n\n"
            "Missing Results:\n" +
            "".join(f"Missing Result {i}:\n{missing}\n\n" for i, missing in enumerate(missing_results, start=1)) +
            "\n" +
            "Extra Results:\n" +
            "".join(f"Extra Result {i}:\n{extra}\n\n" for i, extra in enumerate(extra_results, start=1))
    )

    assert len(results) == len(expected), f"Expected {len(expected)} results, got {len(results)}."


def verify_visitor(visitor: Type['OneByOneVisitor'], source: str, expected: list[str] | str):
    results = visitor(normalize(source)).transform()
    return verify(results, expected)


def verify_transformation(transformation: Type['RegisteredTransformation'], source: str, expected: list[str] | str):
    results = transformation().attack_func(normalize(source))
    return verify(results, expected)
