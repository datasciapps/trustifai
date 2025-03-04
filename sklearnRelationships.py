import inspect
import pkgutil
import importlib
import sklearn
import aif360

def gather_module_classes(package):
    """
    Traverse all submodules of a package and gather information about classes,
    including their fully qualified names, attributes, methods (with arguments
    and return type), and inheritance.

    Returns:
        A dictionary mapping module names to a list of dictionaries, each containing details about a class.
    """
    modules_info = {}
    
    # Traverse all submodules of the package.
    for importer, modname, ispkg in pkgutil.walk_packages(package.__path__, package.__name__ + "."):
        try:
            mod = importlib.import_module(modname)
        except Exception as e:
            print(f"Error importing module {modname}: {e}")
            continue
        
        classes = []
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            # Filter to only include classes from the target package.
            if not obj.__module__.startswith(package.__name__):
                continue
            
            # Fully qualified class name (module path + class name)
            qualified_name = f"{obj.__module__}.{name}"
            
            # Get non-callable attributes (excluding special methods).
            attributes = [
                attr for attr in dir(obj)
                if not callable(getattr(obj, attr, None)) and not attr.startswith("__")
            ]
            
            # Get methods (callable members, excluding dunder methods).
            method_names = [
                method for method in dir(obj)
                if callable(getattr(obj, method, None)) and not method.startswith("__")
            ]
            
            # Collect method details: for each method, store (argument, argument_type) tuples and return type.
            method_details = {}
            for method in method_names:
                try:
                    method_obj = getattr(obj, method)
                    sig = inspect.signature(method_obj)
                    arguments = []
                    for param_name, param_obj in sig.parameters.items():
                        if param_name == "self":
                            continue
                        annotation = param_obj.annotation
                        if annotation is inspect.Parameter.empty:
                            annotation_str = None
                        else:
                            annotation_str = str(annotation)
                        arguments.append((param_name, annotation_str))
                    ret = sig.return_annotation
                    if ret is inspect.Signature.empty:
                        ret = None
                    else:
                        ret = str(ret)
                    method_details[method] = {
                        "arguments": arguments,
                        "return_type": ret
                    }
                except Exception as e:
                    method_details[method] = {"arguments": [], "return_type": None}
            
            is_abstract = inspect.isabstract(obj)
            inheritance = [f"{base.__module__}.{base.__name__}" for base in obj.__bases__]
            
            classes.append({
                "qualified_name": qualified_name,
                "is_abstract": is_abstract,
                "attributes": attributes,
                "methods": method_details,
                "inheritance": inheritance,
                "module": obj.__module__
            })
        
        if classes:
            modules_info[modname] = classes
    return modules_info

def generate_relationship_sentences(modules_info, output_filename):
    """
    Generates textual sentences representing relationships between nodes
    that can be used to create a Neo4j graph.

    Relationships include:
      - For each module: "<module_name> is a module."
      - For each class: "<qualified_class_name> is a class."
      - Class belongs to: "<qualified_class_name> belongs to <module_name>."
      - Class inherits from: "<qualified_class_name> inherits from <base_class>."
      - Class has attribute: "<qualified_class_name> has attribute <attribute>."
      - Class has method: "<qualified_class_name> has method <method>."
      - For each method argument: "<qualified_class_name>.<method> has argument <argument>."
      - For each method argument type: "<qualified_class_name>.<method> argument <argument> has type <argument_type>."
      - Method return type: "<qualified_class_name>.<method> has return type <return_type>."
    """
    with open(output_filename, "w", encoding="utf-8") as f:
        # Output a sentence for each module.
        for modname in sorted(modules_info.keys()):
            f.write(f"{modname} is a module.\n")
        f.write("\n")
        
        for modname, classes in modules_info.items():
            for cls in classes:
                # Output that the class is a class.
                f.write(f"{cls['qualified_name']} is a class.\n")
                # Output BELONGS_TO relationship without the word "module".
                f.write(f"{cls['qualified_name']} belongs to {modname}.\n")
                # Output inheritance relationships.
                for base in cls['inheritance']:
                    f.write(f"{cls['qualified_name']} inherits from {base}.\n")
                # Output attribute relationships.
                for attr in cls['attributes']:
                    f.write(f"{cls['qualified_name']} has attribute {attr}.\n")
                # Output method relationships.
                for method, details in cls['methods'].items():
                    f.write(f"{cls['qualified_name']} has method {method}.\n")
                    # For each argument, output argument and its type.
                    for arg, arg_type in details['arguments']:
                        f.write(f"{cls['qualified_name']}.{method} has argument {arg}.\n")
                        f.write(f"{cls['qualified_name']}.{method} argument {arg} has type {arg_type if arg_type is not None else 'None'}.\n")
                    # Output the method's return type.
                    ret = details['return_type'] if details['return_type'] is not None else "None"
                    f.write(f"{cls['qualified_name']}.{method} has return type {ret}.\n")
                f.write("\n")
    print(f"Relationship sentences have been saved to {output_filename}")

# Gather all classes from scikit-learn (including submodules)
modules_info = gather_module_classes(aif360)

# Generate the relationship sentences and save them to a file.
output_file = "aif360_relationships.txt"
generate_relationship_sentences(modules_info, output_file)