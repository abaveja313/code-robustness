import pytest
from shared.program_utils import remove_comments_and_docstrings, parse_stem


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


def test_with_comments_and_docstrings():
    old_code = '''
def example_function():
    """
    This is a docstring.
    """
    a = 1
    # This is a comment
    b = 2
    c = 3
'''
    new_code = '''
def example_function():
    """
    This is a docstring.
    """
    a = 1
    # This is a comment
    b = 22
    c = 3
'''
    expected_new = '''
def example_function():
    """
    This is a docstring.
    """
    a = 1
    # This is a comment
    b = 22
'''
    expected_old = '''
def example_function():
    """
    This is a docstring.
    """
    a = 1
    # This is a comment
    b = 2
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test with comments and docstrings passed!")


def test_with_inline_comments():
    old_code = '''
def another_function():
    x = 10
    y = 20  # Inline comment
    z = 30
'''
    new_code = '''
def another_function():
    x = 10
    y = 200  # Inline comment
    z = 30a
'''
    expected_new = '''
def another_function():
    x = 10
    y = 200  # Inline comment
'''
    expected_old = '''
def another_function():
    x = 10
    y = 20  # Inline comment
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test with inline comments passed!")


def test_with_block_comments():
    old_code = '''
def function_with_block_comment():
    x = 10
    # Block comment
    y = 20
    z = 30
'''
    new_code = '''
def function_with_block_comment():
    x = 10
    # Block comment
    y = 200
    z = 30
'''
    expected_new = '''
def function_with_block_comment():
    x = 10
    # Block comment
    y = 200
'''
    expected_old = '''
def function_with_block_comment():
    x = 10
    # Block comment
    y = 20
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test with block comments passed!")


def test_without_comments_or_docstrings():
    old_code = '''
def simple_function():
    def nested():
        pass
'''
    new_code = '''
def simple_function():
    def new_nested():
        pass
    
'''
    expected_new = '''
def simple_function():
    def new_nested():
'''
    expected_old = '''
def simple_function():
    def nested():
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test without comments or docstrings passed!")


def test_simple_code_differing_line():
    old_code = '''
def simple_function():
    a = 1
    b = 2
    if a < b:
        c = 3
    else:
        c = 4
    return c
'''
    new_code = '''
def simple_function():
    a = 1
    b = 2
    if a > b:
        c = 3
    else:
        c = 4
    return c
'''
    expected_new = '''
def simple_function():
    a = 1
    b = 2
    if a > b:
'''
    expected_old = '''
def simple_function():
    a = 1
    b = 2
    if a < b:
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test simple code with differing line passed!")


def test_complex_code_differing_line():
    old_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    if a < b:
        x *= 2
    else:
        x /= 2
    return x
'''
    new_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    if a > b:
        x *= 2
    else:
        x /= 2
    return x
'''
    expected_new = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    if a > b:
'''
    expected_old = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    if a < b:
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test complex code with differing line passed!")


def test_docstring_functions_same_docstring_different_lines():
    old_code = '''
def docstring_function():
    """
    This is a docstring.
    """
    a = 1
    b = 2
    return a + b
'''
    new_code = '''
def docstring_function():
    """
    This is a docstring.
    """
    a = 1
    b = 3
    return a + b
'''
    expected_new = '''
def docstring_function():
    """
    This is a docstring.
    """
    a = 1
    b = 3
'''
    expected_old = '''
def docstring_function():
    """
    This is a docstring.
    """
    a = 1
    b = 2
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test with docstring functions, same docstring, but different lines passed!")


def test_single_inline_comment():
    old_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    x += b  # This is an inline comment
    return x
'''
    new_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    x -= b  # This is an inline comment
    return x
'''
    expected_new = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    x -= b  # This is an inline comment
'''
    expected_old = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    x += b  # This is an inline comment
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test single inline comment passed!")


def test_single_block_comment():
    old_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    # This is a block comment
    x += b
    return x
'''
    new_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    # This is a block comment
    x -= b
    return x
'''
    expected_new = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    # This is a block comment
    x -= b
'''
    expected_old = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    # This is a block comment
    x += b
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test single block comment passed!")


def test_multiple_sequential_comments():
    old_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    x += b
    return x
'''
    new_code = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    # Comment 1
    # Comment 3
    x += b
    return x
'''
    expected_new = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    # Comment 1
    # Comment 3
    x += b
'''
    expected_old = '''
def complex_function(x):
    a = 1
    b = 2
    for i in range(5):
        x += i
    x += b
'''
    result_old, result_new = parse_stem(old_code, new_code)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
    print("Test multiple sequential comments passed!")


def test_parse_stem_extra_skips():
    old_code = '''
def func(x):
    a = 1
    print(a)
    print(b)
    '''

    new_code = '''
def func(x):
    a = 1
    a = a
    
    
    print(a)
    print(b)
    '''

    expected_old = '''
def func(x):
    a = 1
    print(a)
    '''

    expected_new = '''
def func(x):
    a = 1
    a = a
    
    
    print(a)
    '''

    result_old, result_new = parse_stem(old_code, new_code, 1)
    assert result_new.strip() == expected_new.strip(), f"Failed for new code. Expected: {expected_new}, Got: {result_new}"
    assert result_old.strip() == expected_old.strip(), f"Failed for old code. Expected: {expected_old}, Got: {result_old}"
