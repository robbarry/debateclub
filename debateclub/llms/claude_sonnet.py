from anthropic import Anthropic
import os
from typing import List, Dict, Any
from pydantic import BaseModel


class ClaudeSonnetModel:
    @staticmethod
    def model_name() -> str:
        return "claude-3-5-sonnet-20241022"

    def __init__(self):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        # Convert messages to Anthropic format
        anthropic_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in messages
        ]
        response = self.client.messages.create(
            model=self.model_name(),
            max_tokens=1024,
            messages=anthropic_messages,
            **kwargs,
        )
        return response.content[0].text
