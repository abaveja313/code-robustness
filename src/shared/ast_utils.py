import ast

from loguru import logger

import ast


def get_function_declaration_line(source_code: str, function_name: str):
    tree = ast.parse(source_code)

    class FunctionVisitor(ast.NodeVisitor):
        def __init__(self):
            self.function_declaration = None

        def visit_FunctionDef(self, node):
            if node.name == function_name:
                params = []

                for arg in node.args.args:
                    params.append(arg.arg)

                if node.args.defaults:
                    defaults = [None] * (len(node.args.args) - len(node.args.defaults)) + node.args.defaults
                    for i, default in enumerate(defaults):
                        if default is not None:
                            params[i] += f"={ast.unparse(default).strip()}"

                if node.args.vararg:
                    params.append(f"*{node.args.vararg.arg}")

                for kwonlyarg, kw_default in zip(node.args.kwonlyargs, node.args.kw_defaults):
                    if kw_default is not None:
                        params.append(f"{kwonlyarg.arg}={ast.unparse(kw_default).strip()}")
                    else:
                        params.append(f"{kwonlyarg.arg}")

                if node.args.kwarg:
                    params.append(f"**{node.args.kwarg.arg}")

                param_str = ", ".join(params)
                self.function_declaration = f"def {function_name}({param_str}):"
            self.generic_visit(node)

    visitor = FunctionVisitor()
    visitor.visit(tree)

    return visitor.function_declaration


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


def has_elif_block(node: ast.If):
    if isinstance(node, ast.If):
        for child in node.orelse:
            if isinstance(child, ast.If):
                return True
            if has_elif_block(child):
                return True
    return False


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
