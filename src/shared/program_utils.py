import ast
import copy
import io
import textwrap
import tokenize
import re

import autopep8
from asttokens import asttokens

IDENT = " " * 4
MUTATED_COMMENT_PREFIX = "# I am a"


class TupleParenthesesTransformer(ast.NodeTransformer):
    def __init__(self, atok):
        self.atok = atok
        self.replacements = []

    def wrap_tuple_with_parens(self, node):
        if isinstance(node, ast.Tuple):
            # Retrieve the source code of the tuple
            tuple_src = self.atok.get_text(node)
            # Wrap the tuple with parentheses
            new_tuple_src = f"({tuple_src})"
            # Save the original and new source code to perform replacement later
            self.replacements.append((self.atok.get_text_range(node), new_tuple_src))

    def visit_Assign(self, node):
        self.generic_visit(node)
        for target in node.targets:
            self.wrap_tuple_with_parens(target)
        self.wrap_tuple_with_parens(node.value)
        return node

    def visit_For(self, node):
        self.generic_visit(node)
        self.wrap_tuple_with_parens(node.target)
        return node

    def visit_With(self, node):
        self.generic_visit(node)
        for item in node.items:
            self.wrap_tuple_with_parens(item.optional_vars)
        return node

    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        for arg in node.args.args:
            self.wrap_tuple_with_parens(arg)
        return node

    def visit_Call(self, node):
        self.generic_visit(node)
        for arg in node.args:
            self.wrap_tuple_with_parens(arg)
        return node


def truncate_code_to_last_function(code):
    lines = code.splitlines()
    function_start_pattern = re.compile(r'^\s*def\s+\w+\s*\(')
    last_function_start = None
    function_body_indent = None

    # Find the start of the last top-level function
    for i, line in enumerate(lines):
        if function_start_pattern.match(line) and not line.strip().startswith(' '):
            last_function_start = i

    if last_function_start is not None:
        # Find the indentation of the function body
        for i in range(last_function_start + 1, len(lines)):
            if lines[i].strip():  # First non-empty line after function definition
                function_body_indent = len(lines[i]) - len(lines[i].lstrip())
                break

        # Find the end of the function
        for i in range(last_function_start + 1, len(lines)):
            line = lines[i]
            if line.strip() and len(line) - len(line.lstrip()) <= 0:
                return '\n'.join(lines[:i])

        # If we've reached this point, the function extends to the end of the file
        return '\n'.join(lines)
    else:
        return code  # No function definitions found, return the original code


def autopep8_normalize_ident(code):
    """
    Attempt to fix the indentation of the code using autopep8.
    """
    options = {
        'select': ['E101', 'E111', 'E114', 'E117'],
        'aggressive': 1
    }
    return autopep8.fix_code(code, options=options)


def fix_odd_indents(code):
    lines = code.splitlines()
    fixed_lines = []

    for line in lines:
        indent_count = len(line) - len(line.lstrip())
        if indent_count % 2 != 0:  # If odd number of indents
            fixed_line = ' ' * (indent_count - 1) + line.lstrip()
        else:
            fixed_line = line
        fixed_lines.append(fixed_line)

    return '\n'.join(fixed_lines)


def transform_parenthesis(source_code):
    atok = asttokens.ASTTokens(source_code, parse=True)
    tree = atok.tree
    transformer = TupleParenthesesTransformer(atok)
    transformer.visit(tree)

    # Apply replacements
    new_source_code = copy.deepcopy(atok.text)
    for (start, end), new_tuple_src in sorted(
            transformer.replacements, key=lambda x: x[0][0], reverse=True
    ):
        new_source_code = (
                new_source_code[:start] + new_tuple_src + new_source_code[end:]
        )

    return new_source_code


def program_concat(stem: str, new_code: str) -> str:
    new_code = new_code.lstrip("\n")
    if stem.endswith("\n"):
        return stem + new_code
    return stem + "\n" + new_code


def remove_pass(prompt: str) -> str:
    """
    Remove `pass` statement from the end of the prompt.

    Since AST transformations have to produce valid code, we have to place
    `pass` inside blocks whenever we mutate statement blocks. Before inference, we want
    to remove it.

    Example Input:
        while True:
            pass # must be here otherwise code is invalid

    """
    prompt_lines = prompt.strip().splitlines()
    if len(prompt_lines) > 0 and prompt_lines[-1].strip() == "pass":
        return "\n".join(prompt_lines[:-1])
    return prompt


def normalize_indentation(code: str) -> str:
    code = textwrap.dedent(code)
    code = code.replace("\t", IDENT)
    lines = [line.rstrip() for line in code.splitlines()]

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

    for tok in tokenize.generate_tokens(io_obj.readline):
        token_type, token_string, start, end, line = tok
        start_line, start_col = start
        end_line, end_col = end
        if start_line > last_lineno:
            out += "\n" * (start_line - last_lineno)
            last_col = 0
        if start_col > last_col:
            out += " " * (start_col - last_col)

        # We don't want to remove these comments because they are placed as part of mutations
        if token_type == tokenize.COMMENT:
            if token_string.startswith(MUTATED_COMMENT_PREFIX):
                out += token_string
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


def parse_stem(old_code: str, new_code: str, extra_skips: int = 0):
    old_lines = old_code.splitlines()
    new_lines = new_code.splitlines()
    if old_lines == new_lines:
        return
    old_index = 0
    new_index = 0
    # Skip matching lines at the beginning
    while old_index < len(old_lines) and new_index < len(new_lines):
        old_line = old_lines[old_index].strip()
        new_line = new_lines[new_index].strip()
        # We don't care about extra newlines being inserted (though this shouldn't happen)
        while old_line == "\n" and old_index < len(old_lines):
            old_index += 1
            continue
        while new_line == "\n" and new_index < len(new_lines):
            new_index += 1
            continue
        # Skip comments and docstrings
        if old_line == new_line or old_line.startswith(('"""', "'''", "#")):
            old_index += 1
            new_index += 1
        else:
            break
    # Capture the function body from the new lines until the first difference
    while new_index < len(new_lines) and (
            new_lines[new_index].strip() == ""
            or new_lines[new_index].strip().startswith("#")
            or new_lines[new_index].strip().startswith(('"""', "'''"))
    ):
        new_index += 1
    while old_index < len(old_lines) and (
            old_lines[old_index].strip() == ""
            or old_lines[old_index].strip().startswith("#")
            or old_lines[old_index].strip().startswith(('"""', "'''"))
    ):
        old_index += 1
    # Ensure capturing the last line if the loop ended due to different lines
    if new_index < len(new_lines):
        new_index += 1

    if old_index < len(old_lines):
        old_index += 1
    # Move new_index to the next extra_skipth line after the difference
    if extra_skips > 0:
        skip_count = 0
        while skip_count < extra_skips and new_index < len(new_lines):
            new_index += 1
            if new_lines[new_index - 1].strip() != "":
                skip_count += 1

    new_func_split = "\n".join(new_lines[:new_index])
    old_func_split = "\n".join(old_lines[:old_index])
    return old_func_split, new_func_split
