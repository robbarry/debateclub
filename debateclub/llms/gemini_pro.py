import google.generativeai as genai
import instructor
import os
from typing import List, Dict, Any, Optional, Type
from pydantic import BaseModel


class GeminiProModel:
    """Gemini Pro model implementation using Instructor for structured outputs."""

    @staticmethod
    def model_name() -> str:
        return "gemini-2.0-flash-exp"

    def __init__(self):
        # Configure the base Gemini client
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        # Create both an Instructor client for structured output and a raw client
        base_model = genai.GenerativeModel("gemini-2.0-flash-exp")
        self.structured_client = instructor.from_gemini(
            client=base_model, mode=instructor.Mode.GEMINI_JSON
        )
        self.raw_client = base_model

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs,
    ) -> Any:
        """Generate a response using Gemini, with optional structured output.

        Args:
            messages: List of conversation messages
            response_model: Optional Pydantic model for structured output
            **kwargs: Additional arguments passed to Gemini's create call

        Returns:
            If response_model is provided, returns an instance of that model.
            Otherwise, returns the raw text response.
        """
        try:
            if response_model:
                # Format messages for Gemini but keep as messages
                return self.structured_client.chat.completions.create(
                    messages=messages, response_model=response_model, **kwargs
                )
            else:
                # For raw text, combine messages and use generate_content
                content = "\n\n".join(msg["content"] for msg in messages)
                response = self.raw_client.generate_content(content, **kwargs)
                return response.text
        except Exception as e:
            print(f"Gemini error: {str(e)}")
            raise
