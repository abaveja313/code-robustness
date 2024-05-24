from .registry import CRT, RegisteredMixin
from .mutation import RegisteredTransformation, OneByOneTransformer
from .visitor import OneByOneVisitor

import importlib
import pkgutil


def load_subclasses():
    """
    Loading subclasses dynamically so they are registered by the metaclasses.
    """
    for _, name, _ in pkgutil.iter_modules(__path__):
        importlib.import_module(f".{name}", __package__)


load_subclasses()
