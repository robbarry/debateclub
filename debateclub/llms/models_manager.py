from typing import Dict, Any
from functools import lru_cache


_loaded_models = None


@lru_cache(maxsize=1)
def get_models() -> Dict[str, Any]:
    """Get or create the singleton dictionary of model instances.

    Returns:
        Dict mapping model names to model instances.
    """
    global _loaded_models
    if _loaded_models is None:
        from debateclub.llms import load_all_models

        _loaded_models = load_all_models()
    return _loaded_models
