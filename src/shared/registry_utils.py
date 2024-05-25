import importlib
import inspect
import pkgutil


def import_subclasses(base_class, module_name):
    subclasses = []

    # Iterate over all modules in the specified package
    package = importlib.import_module(module_name)
    for _, name, is_pkg in pkgutil.walk_packages(package.__path__, package.__name__ + '.'):
        if not is_pkg:
            module = importlib.import_module(name)

            # Iterate over all classes in the module
            for _, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, base_class) and obj != base_class:
                    subclasses.append(obj)

    return subclasses
