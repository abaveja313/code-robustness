import ast


class ASTUtils:
    @staticmethod
    def dfs_walk(node):
        yield node
        for child in ast.iter_child_nodes(node):
            yield from ASTUtils.dfs_walk(child)


class ASTCopier(ast.NodeTransformer):
    def generic_visit(self, node: ast.AST):
        if isinstance(node, ast.AST):
            new_node = type(node)()
            for field, value in ast.iter_fields(node):
                setattr(new_node, field, self.visit(value))
            if hasattr(node, "lineno"):
                new_node.lineno = node.lineno
            if hasattr(node, "col_offset"):
                new_node.col_offset = node.col_offset
            if hasattr(node, "end_lineno"):
                new_node.end_lineno = node.end_lineno
            if hasattr(node, "end_col_offset"):
                new_node.end_col_offset = node.end_col_offset
            return new_node
        elif isinstance(node, list):
            return [self.visit(item) for item in node]
        else:
            return node


class NodeReplacer(ast.NodeTransformer):
    def __init__(self, old_node, new_node):
        self.old_node = old_node
        self.new_node = new_node

    def visit(self, node: ast.AST):
        if isinstance(node, type(self.old_node)) and ast.dump(node) == ast.dump(
                self.old_node
        ):
            return self.new_node
        else:
            return self.generic_visit(node)
