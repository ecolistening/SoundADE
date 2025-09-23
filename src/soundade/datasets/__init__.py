import os
import importlib
import inspect

datasets = {}

package_dir = os.path.dirname(__file__)

# Loop over all .py files in the directory, except __init__.py
for filename in os.listdir(package_dir):
    if filename.endswith('.py') and filename != '__init__.py':
        module_name = filename[:-3]
        full_module_name = f"{__name__}.{module_name}"

        # Dynamically import the module
        module = importlib.import_module(full_module_name)

        # Loop through all classes in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Make sure it's defined in this module (not an import)
            if obj.__module__ == full_module_name:
                datasets[name] = obj
