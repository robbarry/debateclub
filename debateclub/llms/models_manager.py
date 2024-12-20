from typing import Dict
from debateclub.llms import LLMModel
from functools import lru_cache
from . import load_all_models


@lru_cache(maxsize=1)
def get_models() -> Dict[str, LLMModel]:
    """Get or create the singleton dictionary of model instances.

    Returns:
        Dict mapping model names to model instances.
    """
    return load_all_models()
