import inspect
import pkgutil
import importlib
import sklearn
import aif360

def gather_module_classes(package):
    """
    Walk through all submodules of a package and gather information
    about classes, including their attributes, methods (with signatures),
    and inheritance (base classes).

    Returns:
        A dictionary mapping module names to a list of dictionaries, each
        containing details about a class in that module.
    """
    modules_info = {}

    # Traverse all submodules of the package
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            print(f"Error importing module {modname}: {e}")
            continue

        classes = []
        # Get all classes defined in this module
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            # Filter to only include classes from the target package
            if not obj.__module__.startswith(package.__name__):
                continue

            # Get non-callable attributes (excluding special methods)
            attributes = [attr for attr in dir(obj)
                          if not callable(getattr(obj, attr, None)) and not attr.startswith("__")]

            # Get methods (callable members, excluding dunder methods)
            methods = [method for method in dir(obj)
                       if callable(getattr(obj, method, None)) and not method.startswith("__")]

            # Collect method signatures for each method
            method_details = {}
            for method in methods:
                try:
                    method_obj = getattr(obj, method)
                    sig = inspect.signature(method_obj)
                    method_details[method] = str(sig)
                except Exception as e:
                    method_details[method] = "Signature not available"

            # Determine if the class is abstract (to signal potential interfaces)
            is_abstract = inspect.isabstract(obj)

            # Gather inheritance information (list of base classes)
            # Each base is represented as "module.ClassName"
            inheritance = [f"{base.__module__}.{base.__name__}" for base in obj.__bases__]

            classes.append({
                "class_name": name,
                "is_abstract": is_abstract,
                "attributes": attributes,
                "methods": method_details,
                "inheritance": inheritance,
                "module": obj.__module__
            })

        if classes:
            modules_info[modname] = classes
    return modules_info

def save_classes_info_to_file(modules_info, filename):
    """
    Saves the gathered class information (including inheritance details)
    grouped by module to a file.
    """
    with open(filename, "w", encoding="utf-8") as f:
        for modname in sorted(modules_info.keys()):
            f.write(f"Module: {modname}\n")
            f.write("-" * (8 + len(modname)) + "\n")
            for cls in modules_info[modname]:
                kind = "Interface" if cls["is_abstract"] else "Class"
                f.write(f"{kind}: {cls['class_name']}\n")
                f.write(f"  Inherits from: {', '.join(cls['inheritance'])}\n")
                if cls["attributes"]:
                    f.write("  Attributes:\n")
                    for attr in cls["attributes"]:
                        f.write(f"    - {attr}\n")
                if cls["methods"]:
                    f.write("  Methods:\n")
                    for method, signature in cls["methods"].items():
                        f.write(f"    - {method} {signature}\n")
                f.write("\n")  # Blank line between classes
            f.write("=" * 40 + "\n\n")

# Gather all classes from scikit-learn (including submodules)
modules_info = gather_module_classes(aif360)

# Save the detailed textual summary to a file
output_file = "aif360_uml_output.txt"
save_classes_info_to_file(modules_info, output_file)

print(f"UML-like textual information has been saved to {output_file}")