from openai import OpenAI
import os
from typing import List, Dict, Any
from pydantic import BaseModel
import instructor


class GPT4oModel:
    @staticmethod
    def model_name() -> str:
        return "gpt-4o"

    def __init__(self):
        self.client = instructor.patch(OpenAI())

    def generate_response(
        self, messages: List[Dict[str, str]], response_model: type = None, **kwargs
    ) -> Any:
        if response_model:
            response = self.client.chat.completions.create(
                model=self.model_name(),
                response_model=response_model,
                messages=messages,
                **kwargs,
            )
            return response
        else:
            response = self.client.chat.completions.create(
                model=self.model_name(), messages=messages, **kwargs
            )
            return response.choices[0].message.content
