import ast
from abc import abstractmethod, ABC

from asttokens import ASTTokens

from shared.ast_utils import ASTUtils, ASTCopier, NodeReplacer


class OneByOneVisitor(ABC):
    def __init__(self, code: str):
        self.code = code
        self.ast_tree = ASTTokens(self.code, parse=True)
        self.transformations: list[ast.AST] = []
        self.source_code = code

    @property
    def name(self):
        return self.__class__.__name__

    @abstractmethod
    def is_transformable(self, node):
        pass

    @abstractmethod
    def transform_node(self, node) -> list[ast.AST] | ast.AST:
        pass

    def transform(self) -> list[str]:
        for i, node in enumerate(ASTUtils.dfs_walk(self.ast_tree)):
            if i in self.is_transformable(node):
                self.apply_transformations(node)
        return [ast.unparse(tree) for tree in self.transformations]

    def apply_transformations(self, node):
        # Visit a copy of the AST syntax tree
        perturbed_nodes: list[ast.AST] | ast.AST = self.transform_node(
            ASTCopier().visit(node)
        )
        if not isinstance(perturbed_nodes, list):
            perturbed_nodes = [perturbed_nodes]

        for perturbed in perturbed_nodes:
            ast.copy_location(node, perturbed)
            new_tree = ASTCopier().visit(self.ast_tree)
            perturbed_tree = NodeReplacer(node, perturbed).visit(new_tree)
            self.transformations.append(perturbed_tree)
