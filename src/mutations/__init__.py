from .registry import CRT, RegisteredMixin
from .mutation import RegisteredTransformation, OneByOneTransformer
from .visitor import OneByOneVisitor

import importlib
import pkgutil


def load_subclasses(package_name=__name__):
    """
    Loading subclasses dynamically so they are registered by the metaclasses.
    """
    package = importlib.import_module(package_name)
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if not is_pkg:
            importlib.import_module(name)

load_subclasses()