import time
from typing import List, Optional
from pydantic import ValidationError
from debateclub.models import DebateTopic, Position, DebateArgument
from debateclub.llms import load_all_models

models = load_all_models()


def generate_argument(
    model_name: str,
    topic: DebateTopic,
    position: Position,
    previous_arguments: Optional[List[DebateArgument]] = None,
    max_retries=3,
) -> DebateArgument:
    """Generates a debate argument using the specified LLM."""
    last_argument = previous_arguments[0] if previous_arguments else None

    context = f"""Topic: {topic.topic}
    Your position: {position.value}
    Context: {topic.context}
    {'Pro position: ' + topic.pro_position if position == Position.PRO else 'Con position: ' + topic.con_position}

    """
    if last_argument:
        context += f"""
    The opposing side's last argument was:
        Introduction: {last_argument.introduction}
        Reasoning: {[premise.premise for premise in last_argument.reasoning]}
        Rebuttal: {last_argument.rebuttal}

    Craft a detailed and comprehensive argument in support of your position that directly responds to the opposition’s last argument.
    Your argument should be well-reasoned, and follow the required structure, including supporting key points with a chain of reasoning, and anticipating counter arguments to the last point made by your opponent.
    """
    else:
        context += """
        Provide a detailed and comprehensive argument in support of your position.
    Your argument should be well-reasoned and should follow the required structure, including supporting key points with a chain of reasoning, and anticipating counter arguments to your position.
    """

    context += f"""
    Provide your response in the following format:
    {{
        "position": "{position.value}",
        "introduction": "A concise 1-2 sentence introduction to your argument",
        "reasoning": [
            {{"premise": "Premise 1", "reasoning": "Reasoning chain for Premise 1"}},
            {{"premise": "Premise 2", "reasoning": "Reasoning chain for Premise 2"}},
            {{"premise": "Premise 3", "reasoning": "Reasoning chain for Premise 3"}}
            ],
        "rebuttal": "A direct rebuttal to the opponent's previous key points"
    }}
    """

    for attempt in range(max_retries):
        try:
            response = _create_completion(
                model_name,
                [{"role": "user", "content": context}],
                DebateArgument,
                is_json=True,
            )
            response.position = (
                position  # overwrite the LLM response as it could be wrong still
            )

            return response
        except ValidationError as e:
            if attempt < max_retries - 1:  # Don't wait on the last attempt
                wait_time = 2**attempt  # Exponential backoff
                print(f"Validation Error: {e}. Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
            else:
                print(
                    f"Validation Error: {e}. Max retries exceeded. Failed to create argument."
                )
                raise  # re-raise the error for logging


def _create_completion(
    model_name: str,
    messages: List[dict],
    response_model: type = None,
    is_json=True,
) -> any:
    if model_name not in models:
        raise ValueError(
            f"Model '{model_name}' not found: Available models are {list(models.keys())}"
        )

    model = models[model_name]
    if model_name == "claude-3-5-sonnet-20241022":
        response_text = model().generate_response(messages)
    elif model_name == "gemini-exp-1206":
        response_text = model().generate_response(messages)
    else:
        response_text = model().generate_response(
            messages, response_model=response_model
        )

    # Basic sanitization of the response
    if isinstance(response_text, str):
        text = "".join(ch for ch in response_text if 0x20 <= ord(ch) < 0x10000)
    elif hasattr(response_text, "model_dump_json"):  # Handle Pydantic models
        text = response_text.model_dump_json()
    else:
        text = str(response_text)

    if is_json:
        try:
            return response_model.model_validate_json(text)
        except ValidationError as e:
            print(f"JSON Validation Error: {e}")
            print(f"Problematic JSON: {text}")
            raise
    else:
        return response_model(text=text)
