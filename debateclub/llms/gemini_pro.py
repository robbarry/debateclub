import google.generativeai as genai
import os
from typing import List, Dict, Any
from pydantic import BaseModel


class GeminiProModel:
    @staticmethod
    def model_name() -> str:
        return "gemini-2.0-flash-exp"

    def __init__(self):
        genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
        self.client = genai.GenerativeModel("gemini-2.0-flash-exp")

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        content = "\n\n".join(msg["content"] for msg in messages)
        response = self.client.generate_content(content, **kwargs)
        if hasattr(response, "text"):
            return response.text
        else:
            raise ValueError(f"Gemini response did not contain text: {response}")
