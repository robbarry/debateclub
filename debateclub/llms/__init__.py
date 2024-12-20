import os
import importlib.util
import inspect
from typing import Dict, Type


def load_all_models() -> Dict[str, Type]:
    """
    Loads all LLM classes from the llms directory.

    Returns:
        A dictionary mapping model names to model classes.
    """
    models = {}
    llms_dir = os.path.dirname(__file__)
    for filename in os.listdir(llms_dir):
        if filename.endswith(".py") and filename != "__init__.py":
            module_name = f"debateclub.llms.{filename[:-3]}"
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(llms_dir, filename)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and hasattr(obj, "model_name"):
                    try:
                        models[obj.model_name()] = obj
                    except Exception as e:
                        print(f"Error loading model {name} from {filename}: {e}")
    return models
