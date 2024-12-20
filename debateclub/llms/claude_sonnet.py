from anthropic import Anthropic
import instructor
import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel


class ClaudeSonnetModel:
    """Claude 3.5 Sonnet model implementation using Instructor for structured outputs."""

    @staticmethod
    def model_name() -> str:
        return "claude-3-5-sonnet-20241022"

    def __init__(self):
        # Initialize the Anthropic client with Instructor patching
        client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.client = instructor.from_anthropic(client)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Any:
        """Generate a response using Claude, with optional structured output.

        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model for structured output
            **kwargs: Additional arguments passed to Anthropic's create call

        Returns:
            If response_model is provided, returns an instance of that model.
            Otherwise, returns the raw text response.
        """
        # Standard message format for Anthropic
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]

        if response_model:
            # Use Instructor's structured output handling
            response = self.client.messages.create(
                model=self.model_name(),
                messages=anthropic_messages,
                response_model=response_model,
                max_tokens=1024,
                **kwargs,
            )
            return response
        else:
            # Handle raw text response
            response = self.client.messages.create(
                model=self.model_name(),
                messages=anthropic_messages,
                max_tokens=1024,
                **kwargs,
            )
            return response.content[0].text
