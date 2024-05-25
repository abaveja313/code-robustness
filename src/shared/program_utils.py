import ast
import copy
import io
import textwrap
import tokenize
from typing import List


def remove_pass(*, inputs: List[str]) -> List[str]:
    results = []
    for prompt in inputs:
        prompt_lines = prompt.strip().splitlines()
        if len(prompt_lines) > 0 and prompt_lines[-1].strip() == "pass":
            results.append("\n".join(prompt_lines[:-1]))
        else:
            results.append(prompt)

    return results


def normalize_indentation(code: str) -> str:
    code = textwrap.dedent(code)
    # Split the input code into lines and strip trailing whitespace
    lines = [line.rstrip() for line in code.splitlines()]

    # Remove leading blank lines
    while lines and not lines[0].strip():
        lines.pop(0)

    leading_spaces = [len(line) - len(line.lstrip()) for line in lines if line.strip()]
    if not leading_spaces:
        return ""

    unique_leading_spaces = sorted(set(leading_spaces))
    a = unique_leading_spaces[0]
    b = unique_leading_spaces[1] if len(unique_leading_spaces) > 1 else a

    def normalize_line(line: str) -> str:
        stripped_line = line.lstrip()
        leading_space_count = len(line) - len(stripped_line)

        if leading_space_count == 0:
            return line

        if leading_space_count == a:
            normalized_spaces = 0
        else:
            normalized_spaces = (leading_space_count // b) * 4

        return " " * normalized_spaces + stripped_line

    if a != 0:
        lines = [line[a:] if line.startswith(" " * a) else line for line in lines]

    normalized_lines = [normalize_line(line) for line in lines]
    return "\n".join(normalized_lines)


def remove_comments_and_docstrings(source, remove_docstrings=False):
    """
    Remove comments and optionally docstrings from a Python code string,
    while preserving indentation and spacing correctly.
    """
    io_obj = io.StringIO(source)
    out = ""
    prev_toktype = tokenize.INDENT
    last_lineno = -1
    last_col = 0
    first_token = True

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type, token_string, start, end, line = tok
        start_line, start_col = start
        end_line, end_col = end

        if start_line > last_lineno:
            out += "\n" * (start_line - last_lineno)
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)

        if token_type == tokenize.COMMENT:
            pass
        elif (
                token_type == tokenize.STRING
                and remove_docstrings
                and prev_toktype == tokenize.INDENT
        ):
            pass
        else:
            out += token_string

        prev_toktype = token_type
        last_col = end_col
        last_lineno = end_line

    cleaned_lines = [line.rstrip() for line in out.splitlines() if line.strip()]
    if cleaned_lines:
        base_indentation = len(cleaned_lines[0]) - len(cleaned_lines[0].lstrip())
        cleaned_lines = [
            (line[base_indentation:] if len(line) > base_indentation else line)
            for line in cleaned_lines
        ]
    return "\n".join(cleaned_lines)


def extract_function_parts(code: str, function_name: str = None):
    parsed_code = ast.parse(code)
    func_node = None
    for node in parsed_code.body:
        if isinstance(node, ast.FunctionDef) and (
                function_name is None or node.name == function_name
        ):
            func_node = node
            break

    if func_node is None:
        raise ValueError(f"Function '{function_name}' not found in the provided code.")

    func_start_line = func_node.lineno - 1
    func_end_line = func_node.end_lineno
    func_code_lines = code.splitlines()

    func_declaration_line = func_code_lines[func_start_line].strip()
    docstring = ast.get_docstring(func_node)
    if docstring:
        docstring_lines = docstring.splitlines()
        indented_docstring = "\n".join(["    " + line for line in docstring_lines])
        indented_docstring = f'    """{indented_docstring}\n    """'  # Properly format and indent multiline docstrings
    else:
        indented_docstring = ""

    func_body_with_comments = "\n".join(
        func_code_lines[func_start_line + 1: func_end_line]
    )
    func_body = remove_comments_and_docstrings(
        func_body_with_comments, remove_docstrings=True
    )
    func_body = "\n    ".join(
        func_body.splitlines()
    )  # Correctly indent all lines of the body

    function_parts = {
        "declaration": func_declaration_line
                       + ("\n" + indented_docstring if indented_docstring else ""),
        "body": "    " + func_body.replace("\n\n", "\n"),  # Add initial indentation
    }
    return function_parts


def extract_function_parts(code: str, function_name: str = None):
    parsed_code = ast.parse(code)
    func_node = None
    for node in parsed_code.body:
        if isinstance(node, ast.FunctionDef) and (
                function_name is None or node.name == function_name
        ):
            func_node = node
            break

    if func_node is None:
        raise ValueError(f"Function '{function_name}' not found in the provided code.")

    func_start_line = func_node.lineno - 1
    func_end_line = func_node.end_lineno
    func_code_lines = code.splitlines()

    func_declaration_line = func_code_lines[func_start_line].strip()
    docstring = ast.get_docstring(func_node)
    if docstring:
        docstring_lines = docstring.splitlines()
        indented_docstring = "\n".join(["    " + line for line in docstring_lines])
        indented_docstring = f'    """{indented_docstring}\n    """'  # Properly format and indent multiline docstrings
    else:
        indented_docstring = ""

    func_body_with_comments = "\n".join(
        func_code_lines[func_start_line + 1: func_end_line]
    )
    func_body = remove_comments_and_docstrings(
        func_body_with_comments, remove_docstrings=True
    )
    func_body = "\n    ".join(
        func_body.splitlines()
    )  # Correctly indent all lines of the body

    function_parts = {
        "declaration": func_declaration_line
                       + ("\n" + indented_docstring if indented_docstring else ""),
        "body": "    " + func_body.replace("\n\n", "\n"),  # Add initial indentation
    }
    return function_parts


def parse_stem(old_code: str, new_code: str, function_name: str = None):
    old_parts = extract_function_parts(old_code, function_name=function_name)
    new_parts = extract_function_parts(new_code, function_name=function_name)
    old_lines = old_parts["body"].splitlines()
    new_lines = new_parts["body"].splitlines()

    for i, (old_line, new_line) in enumerate(zip(old_lines, new_lines)):
        if old_line != new_line:
            break
    else:
        return f"{new_parts['declaration']}\n{new_parts['body']}", f"{old_parts['declaration']}\n{old_parts['body']}"

    if (
            i == len(old_lines) - 1
            and i == len(new_lines) - 1
            and old_lines[i] == new_lines[i]
    ):
        return f"{new_parts['declaration']}\n{new_parts['body']}", f"{old_parts['declaration']}\n{old_parts['body']}"

    new_func_split = "\n".join(new_lines[: i + 1])
    old_func_split = "\n".join(old_lines[: i + 1])

    return f"{new_parts['declaration']}\n{new_func_split}", f"{old_parts['declaration']}\n{old_func_split}"


def one_by_one(key: str, obj: object):
    if not hasattr(obj, key):
        raise ValueError(f"Object {obj} has no field {key}")

    for idx, value in enumerate(getattr(obj, key)):
        new_obj = copy.deepcopy(obj)
        yield new_obj, getattr(new_obj, key)[idx]

expected = [
            """
            def doSomething(a, b):
                if name == 'foo':  # I am a comment
                    doSomething(1, 2)
                return a + b
            """,
            """
            def doSomething(a, b):
                if name == 'foo':
                    doSomething(1, 2)  # I am a comment
                return a + b
            """,
            """
            def doSomething(a, b):
                if name == 'foo':
                    doSomething(1, 2)
                return a + b  # I am a comment
            """
        ]
