import ast
import copy
import uuid
from abc import abstractmethod, ABC

from asttokens import ASTTokens

from shared.ast_utils import dfs_walk, NodeReplacer, StatementGroupExpander


class OneByOneVisitor(ABC):
    def __init__(self, code: str):
        self.code = code
        self.ast_tokens = ASTTokens(self.code, parse=True)
        self.ast_tree = self.ast_tokens.tree
        self.transformations: list[ast.AST] = []
        self.source_code = code

        self._add_node_metadata(self.ast_tree)

    def _add_node_metadata(self, node: ast.AST):
        node.uuid = uuid.uuid4().hex
        for child in ast.iter_child_nodes(node):
            child.parent = node
            self._add_node_metadata(child)

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
        for i, node in enumerate(dfs_walk(self.ast_tree)):
            if self.is_transformable(node):
                self.apply_transformations(node)
        return [ast.unparse(tree) for tree in self.transformations]

    def apply_transformations(self, node):
        # Transform a copy of the node to avoid modifying the original tree
        # It is possible for a single node transformation to return multiple perturbations
        perturbed_nodes: list[ast.AST] | ast.AST = self.transform_node(
            copy.deepcopy(node)
        )
        if not isinstance(perturbed_nodes, list):
            perturbed_nodes = [perturbed_nodes]

        for perturbed in perturbed_nodes:
            ast.copy_location(node, perturbed)
            new_tree = copy.deepcopy(self.ast_tree)
            perturbed_tree = NodeReplacer(node, perturbed).visit(new_tree)

            # This expands basic blocks created by the transformation into the parent node's body
            inlined_tree = StatementGroupExpander().visit(perturbed_tree)
            inlined_tree = ast.fix_missing_locations(inlined_tree)
            self.transformations.append(inlined_tree)
