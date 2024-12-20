import google.generativeai as genai
import os
from typing import List, Dict, Any
from pydantic import BaseModel


class GeminiProModel:
    @staticmethod
    def model_name() -> str:
        return "gemini-exp-1206"

    def __init__(self):
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.client = genai.GenerativeModel("gemini-exp-1206")

    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        content = "\n\n".join(msg["content"] for msg in messages)
        response = self.client.generate_content(content, **kwargs)
        if hasattr(response, "text"):
            return response.text
        else:
            raise ValueError(f"Gemini response did not contain text: {response}")
