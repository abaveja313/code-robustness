import textwrap
from typing import Type

from mutations import OneByOneVisitor


def normalize(src):
    src = textwrap.dedent(src)
    return src.strip()


def verify_visitor(visitor: Type['OneByOneVisitor'], source: str, expected: list[str] | str):
    results = visitor(normalize(source)).transform()
    if not isinstance(expected, list):
        expected = [expected]
    expected = list(map(normalize, expected))
    for r in results:
        print(r)
    assert len(results) == len(expected), f"Expected {len(expected)} results, got {len(results)}"

    for i, (result, exp) in enumerate(zip(results, expected)):
        assert result == exp, f"Expected at index {i}:\n{repr(exp)}\nGot:\n{repr(result)}"
