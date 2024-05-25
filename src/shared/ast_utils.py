import ast

from loguru import logger

import ast


class ParentTransformer(ast.NodeTransformer):
    def visit(self, node):
        for child in ast.iter_child_nodes(node):
            child.parent = node
        return self.generic_visit(node)


class StatementGroup(ast.AST):
    _fields = ['body']

    def __init__(self, body=None):
        self.body = body or []
        super().__init__()


class StatementGroupExpander(ast.NodeTransformer):
    def visit_StatementGroup(self, node):
        module = ast.Module(body=node.body)
        module.type_ignores = []
        return module


def dfs_walk(node):
    yield node
    for child in ast.iter_child_nodes(node):
        yield from dfs_walk(child)


def is_unary_assign(node):
    return isinstance(node, ast.Assign) and not isinstance(node.targets[0], ast.Tuple)


def get_constant_num(node):
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
        return ast.Constant(value=-1 * node.operand.value)
    return node


class NodeReplacer(ast.NodeTransformer):
    def __init__(self, node1, node2):
        self.node1 = node1
        self.node2 = node2

    @staticmethod
    def get_nid(node):
        node_uuid = getattr(node, 'uuid')
        if not node_uuid:
            logger.warning(f"Found node {node} without `uuid` attribute")
        return node_uuid

    def visit(self, node):
        if self.get_nid(node) == self.get_nid(self.node1):
            return self.node2
        return self.generic_visit(node)
