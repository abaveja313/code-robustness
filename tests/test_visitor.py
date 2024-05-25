import ast
import copy

from mutations.visitor import OneByOneVisitor

from utils import verify_visitor


class DummyOneByOneVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.Assign)

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        node.value = ast.Constant(value=42)
        return node


class MultiOneByOneVisitor(OneByOneVisitor):
    def is_transformable(self, node):
        return isinstance(node, ast.Assign) and len(node.targets) == 1

    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        res = []
        for i in range(3):
            node_copy = copy.deepcopy(node)
            node_copy.targets[0].id = f"var_{i}"
            res.append(node_copy)

        return res


def test_name():
    visitor = DummyOneByOneVisitor("")
    assert visitor.name == "DummyOneByOneVisitor"


def test_one_by_one_visitor_same_dumps():
    code = """
    # something
    a = 1
    a = 1
    """

    expected_transforms = ["""a = 42\na = 1""", """a = 1\na = 42"""]
    verify_visitor(DummyOneByOneVisitor, code, expected_transforms)


def test_one_by_one_visitor_same_line():
    code = """
    a = True;a=False
    """
    expected_transforms = ["""a = 42\na = False""", """a = True\na = 42"""]
    verify_visitor(DummyOneByOneVisitor, code, expected_transforms)


def test_multi_visitor():
    code = """
    foo = 2
    print(foo)
    """
    expected = ["var_0 = 2\nprint(foo)", "var_1 = 2\nprint(foo)", "var_2 = 2\nprint(foo)"]
    verify_visitor(MultiOneByOneVisitor, code, expected)
