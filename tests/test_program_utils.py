import pytest
from shared.program_utils import remove_comments_and_docstrings, extract_function_parts, parse_stem


def test_remove_comments():
    source = '''# This is a comment
def foo():
    # Another comment
    return 42  # Trailing comment
'''
    expected = '''def foo():
    return 42'''
    assert remove_comments_and_docstrings(source) == expected


def test_remove_docstrings():
    source = '''def foo():
    """
    This is a docstring
    """
    return 42
'''
    expected = '''def foo():
    return 42'''
    assert remove_comments_and_docstrings(source, remove_docstrings=True) == expected


def test_preserve_non_docstring_strings():
    source = '''def foo():
    x = "This is a string"
    return x
'''
    expected = '''def foo():
    x = "This is a string"
    return x'''
    assert remove_comments_and_docstrings(source, remove_docstrings=True) == expected


def test_mixed_content():
    source = '''# Initial comment
def foo():
    """
    Docstring for foo
    """
    # Another comment
    x = "This is a string"  # Trailing comment
    return x
'''
    expected = '''def foo():
    x = "This is a string"
    return x'''
    assert remove_comments_and_docstrings(source, remove_docstrings=True) == expected


def test_comment_with_specific_prefix():
    source = '''# I am a special comment
def foo():
    # I am a special comment
    return 42
'''
    expected = '''# I am a special comment
def foo():
    # I am a special comment
    return 42'''
    assert remove_comments_and_docstrings(source) == expected


def test_extract_function_parts_basic():
    code = '''
def foo():
    return 42
'''
    expected = {
        "declaration": "def foo():",
        "body": "    return 42"
    }
    assert extract_function_parts(code, "foo") == expected


def test_extract_function_parts_with_docstring():
    code = '''
def foo():
    """
    This is a docstring
    """
    return 42
'''
    expected = {
        "declaration": 'def foo():\n    """\n    This is a docstring\n    """',
        "body": "    return 42"
    }
    assert extract_function_parts(code, "foo") == expected


def test_extract_function_parts_with_comments():
    code = '''
def foo():
    # This is a comment
    return 42  # Trailing comment
'''
    expected = {
        "declaration": "def foo():",
        "body": "    return 42"
    }
    assert extract_function_parts(code, "foo") == expected


def test_extract_function_parts_multiple_functions():
    code = '''
def foo():
    return 42

def bar():
    return 84
'''
    expected_foo = {
        "declaration": "def foo():",
        "body": "    return 42"
    }
    expected_bar = {
        "declaration": "def bar():",
        "body": "    return 84"
    }
    assert extract_function_parts(code, "foo") == expected_foo
    assert extract_function_parts(code, "bar") == expected_bar


def test_function_not_found():
    code = '''
def foo():
    return 42
'''
    with pytest.raises(ValueError, match="Function 'bar' not found in the provided code."):
        extract_function_parts(code, "bar")


def test_parse_stem_identical_functions():
    old_code = '''
def foo():
    return 42
'''
    new_code = '''
def foo():
    return 42
'''
    expected = (
        'def foo():\n    return 42',
        'def foo():\n    return 42'
    )
    assert parse_stem(old_code, new_code, "foo") == expected


def test_parse_stem_different_functions():
    old_code = '''
def foo():
    return 42
'''
    new_code = '''
def foo():
    return 84
'''
    expected = (
        'def foo():\n    return 84',
        'def foo():\n    return 42'
    )
    assert parse_stem(old_code, new_code, "foo") == expected


def test_parse_stem_partial_match():
    old_code = '''
def foo():
    a = 1
    return 42
'''
    new_code = '''
def foo():
    a = 1
    return 84
'''
    expected = (
        'def foo():\n    a = 1\n    return 84',
        'def foo():\n    a = 1\n    return 42'
    )
    assert parse_stem(old_code, new_code, "foo") == expected


def test_parse_stem_function_not_found():
    old_code = '''
def foo():
    return 42
'''
    new_code = '''
def bar():
    return 84
'''
    with pytest.raises(ValueError, match="Function 'foo' not found in the provided code."):
        parse_stem(old_code, new_code, "foo")


if __name__ == "__main__":
    pytest.main()
