import os
import importlib.util
import inspect
from typing import Dict, Protocol, List, Optional, Any, Type
from pydantic import BaseModel


class LLMModel(Protocol):
    """Protocol defining the interface for LLM models."""

    @staticmethod
    def model_name() -> str:
        """Returns the identifier for this model."""
        ...

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Any:
        """Generates a response from the model."""
        ...


def load_all_models() -> Dict[str, Any]:
    """Loads and instantiates all LLM model implementations."""
    models = {}
    llms_dir = os.path.dirname(__file__)

    print("\nLoading LLM models:")
    for filename in os.listdir(llms_dir):
        if (
            filename.endswith(".py")
            and filename != "__init__.py"
            and filename != "models_manager.py"
        ):
            module_name = f"debateclub.llms.{filename[:-3]}"
            spec = importlib.util.spec_from_file_location(
                module_name, os.path.join(llms_dir, filename)
            )
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            for name, obj in inspect.getmembers(module):
                if (
                    inspect.isclass(obj)
                    and hasattr(obj, "model_name")
                    and hasattr(obj, "generate_response")
                    and not isinstance(obj, Protocol)
                ):
                    try:
                        model_instance = obj()
                        models[obj.model_name()] = model_instance
                        print(f"✓ Successfully loaded {obj.model_name()}")
                    except Exception as e:
                        print(f"Error loading model {name} from {filename}: {e}")

    return models
