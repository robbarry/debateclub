from openai import OpenAI
import instructor
import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel


class GPT4oModel:
    """GPT-4o model implementation using Instructor for structured outputs."""

    @staticmethod
    def model_name() -> str:
        return "gpt-4o"

    def __init__(self):
        # Initialize OpenAI client with Instructor patching
        client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.client = instructor.from_openai(client)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Any:
        """Generate a response using GPT-4o, with optional structured output.

        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model for structured output
            **kwargs: Additional arguments passed to OpenAI's create call

        Returns:
            If response_model is provided, returns an instance of that model.
            Otherwise, returns the raw text response.
        """
        if response_model:
            # Use Instructor's structured output handling
            response = self.client.chat.completions.create(
                model="gpt-4o",  # Actual model name for API calls
                messages=messages,
                response_model=response_model,
                **kwargs,
            )
            return response
        else:
            # Handle raw text response
            response = self.client.chat.completions.create(
                model="gpt-4o", messages=messages, **kwargs
            )
            return response.choices[0].message.content
