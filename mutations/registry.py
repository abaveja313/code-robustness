from abc import ABCMeta
from enum import Enum, auto

from loguru import logger


class CRT(Enum):
    code_style = auto()
    numbers = auto()
    math = auto()
    strings = auto()
    booleans = auto()
    arrays = auto()
    dicts = auto()
    conditionals = auto()
    loops = auto()


class MutationRegistry:
    _registry = {}

    @classmethod
    def register(cls, subclass, category):
        logger.info(f"Registered Mutation {subclass.__name__} to category {category}")
        cls._registry.setdefault(category, [])
        cls._registry[category].append(subclass)

    @classmethod
    def get(cls, category=None, exclude=None):
        if category and exclude:
            raise ValueError('Cannot provide both category and exclude')
        if not (category or exclude):
            return list(cls._registry.values())
        if category:
            return cls._registry.get(category, [])
        else:
            valid = set(cls._registry.keys()) - set(exclude)
            return [subclass for key, values in cls._registry.items() if key in valid for subclass in values]


class RegisteredMeta(ABCMeta):
    def __new__(cls, name, bases, attrs, **kwargs):
        new_cls = super().__new__(cls, name, bases, attrs)
        if not kwargs.get('abstract', False):
            category = kwargs.get('category')
            if category is None:
                raise ValueError(f"Category must be provided for registered class {name}")

            MutationRegistry.register(new_cls, category)
        return new_cls


class RegisteredMixin(metaclass=RegisteredMeta, abstract=True):
    pass
