import os
import importlib.util
import inspect
from typing import Dict, Protocol, List, Optional, Any, Type
from pydantic import BaseModel


class LLMModel(Protocol):
    """Protocol defining the interface for LLM models.

    All LLM implementations should follow this protocol to ensure consistent
    behavior across different providers. This protocol assumes the use of
    Instructor for handling structured outputs.

    Methods:
        model_name: Returns the identifier for this model
        generate_response: Generates a response from the model, optionally parsed into a Pydantic model
    """

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
        """Generates a response from the model.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            response_model: Optional Pydantic model to parse the response into
            **kwargs: Additional arguments passed to the underlying API

        Returns:
            If response_model is provided, returns an instance of that model.
            Otherwise, returns the raw text response.
        """
        ...


def load_all_models() -> Dict[str, LLMModel]:
    """Loads and instantiates all LLM model implementations.

    Note: Will print debug information about loaded models and any errors.

    Dynamically loads all Python files in the llms directory (except __init__.py)
    and instantiates any classes that implement the LLMModel protocol.

    Returns:
        Dictionary mapping model names to model instances.
    """
    models = {}
    llms_dir = os.path.dirname(__file__)

    print("\nLoading LLM models:")
    for filename in os.listdir(llms_dir):
        if filename.endswith(".py") and filename != "__init__.py":
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
                ):
                    try:
                        model_instance = obj()
                        models[obj.model_name()] = model_instance
                        print(f"✓ Successfully loaded {obj.model_name()}")
                    except Exception as e:
                        print(f"Error loading model {name} from {filename}: {e}")

    return models
