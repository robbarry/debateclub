from typing import List, Tuple
from debateclub.models import (
    DebateTopic,
    DebateArgument,
    JudgmentExtraction,
    JudgmentResult,
    Position,
    TextResponse,
)
from debateclub.llms import load_all_models
from pydantic import ValidationError
import re


models = load_all_models()


def judge_debate(
    judge_model_name: str,
    topic: DebateTopic,
    pro_arguments: List[DebateArgument],
    con_arguments: List[DebateArgument],
) -> JudgmentResult:
    """Judges the debate based on the provided arguments."""

    debate_summary = f"""Topic: {topic.topic}
    Context: {topic.context}

    Pro Position Arguments:
    {_format_arguments(pro_arguments)}

    Con Position Arguments:
    {_format_arguments(con_arguments)}

    Please extract the claims and supporting reasoning from each side, and identify any logical fallacies present in the debate.
    Provide your response in the following JSON format:
    {{
        "pro_claims": [
            {{"claim": "Claim 1", "reasoning": "Reasoning 1"}},
            {{"claim": "Claim 2", "reasoning": "Reasoning 2"}}, ...
        ],
        "con_claims": [
            {{"claim": "Claim 1", "reasoning": "Reasoning 1"}},
            {{"claim": "Claim 2", "reasoning": "Reasoning 2"}}, ...
        ],
        "pro_rebuttals": ["Rebuttal 1", "Rebuttal 2", ...],
        "con_rebuttals": ["Rebuttal 1", "Rebuttal 2", ...],
        "logical_fallacies": ["Fallacy 1", "Fallacy 2", ...]
    }}
    """

    extraction = _create_completion(
        judge_model_name,
        [{"role": "user", "content": debate_summary}],
        JudgmentExtraction,
    )
    # transform dict to tuple.
    if extraction.pro_claims:
        extraction.pro_claims = [
            (item["claim"], item["reasoning"]) for item in extraction.pro_claims
        ]
    if extraction.con_claims:
        extraction.con_claims = [
            (item["claim"], item["reasoning"]) for item in extraction.con_claims
        ]

    scoring_prompt = f"""
    Based on the extracted information:

    Topic: {topic.topic}
    Context: {topic.context}

    Pro Claims: {extraction.pro_claims}
    Con Claims: {extraction.con_claims}
    Pro Rebuttals: {extraction.pro_rebuttals}
    Con Rebuttals: {extraction.con_rebuttals}
    Logical Fallacies: {extraction.logical_fallacies}

    Judge the debate based on the extracted information, following these steps:
        1. Identify explicit logical fallacies, if any, and deduct points for each.
        2. Assess the quality and completeness of the reasoning chains provided by each side.
        3. Assess how directly each side addressed the opponent's key points in their rebuttal.

    Provide your judgment in the following format:
    {{
        "winner": "pro or con",
        "pro_score": {{
            "logic_score": (integer between 1-10),
            "evidence_score": (integer between 1-10),
            "rebuttal_score": (integer between 1-10),
            "overall_score": (integer between 1-10),
            "reasoning": "Explanation of scores"
        }},
        "con_score": {{
            "logic_score": (integer between 1-10),
            "evidence_score": (integer between 1-10),
            "rebuttal_score": (integer between 1-10),
            "overall_score": (integer between 1-10),
            "reasoning": "Explanation of scores"
        }},
        "explanation": "Detailed explanation of your decision"
    }}
    """

    response = _create_completion(
        judge_model_name,
        [{"role": "user", "content": scoring_prompt}],
        JudgmentResult,
    )

    # convert to ints incase LLM failed to follow instructions
    response.pro_score.logic_score = int(response.pro_score.logic_score)
    response.pro_score.evidence_score = int(response.pro_score.evidence_score)
    response.pro_score.rebuttal_score = int(response.pro_score.rebuttal_score)
    response.pro_score.overall_score = int(response.pro_score.overall_score)

    response.con_score.logic_score = int(response.con_score.logic_score)
    response.con_score.evidence_score = int(response.con_score.evidence_score)
    response.con_score.rebuttal_score = int(response.con_score.rebuttal_score)
    response.con_score.overall_score = int(response.con_score.overall_score)

    return response


def summarize_topic(judge_model_name: str, topic: DebateTopic) -> str:
    """Summarizes the topic using the specified LLM."""
    prompt = f"""Please provide a short, high-level summary of the following debate topic.

      Topic: {topic.topic}
      Context: {topic.context}
      Pro Position: {topic.pro_position}
      Con Position: {topic.con_position}
      """
    response = _create_completion(
        judge_model_name,
        [{"role": "user", "content": prompt}],
        TextResponse,
        is_json=False,
    )
    return response.text


def _format_arguments(arguments: List[DebateArgument]) -> str:
    """Formats a list of debate arguments into a readable string."""
    formatted_args = []
    for i, arg in enumerate(arguments):
        formatted_arg = f"Round {i + 1}:\n"
        formatted_arg += f"  Introduction: {arg.introduction}\n"
        formatted_arg += "  Reasoning:\n"
        for j, premise in enumerate(arg.reasoning):
            formatted_arg += (
                f"    {j+1}. {premise.premise} (Reasoning: {premise.reasoning})\n"
            )
        formatted_arg += f"  Rebuttal: {arg.rebuttal}\n"
        formatted_args.append(formatted_arg)
    return "\n".join(formatted_args)


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
        # Remove code blocks
        text = re.sub(r"```json\s*(.*?)\s*```", r"\1", response_text, flags=re.DOTALL)
        text = "".join(ch for ch in text if 0x20 <= ord(ch) < 0x10000)
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
