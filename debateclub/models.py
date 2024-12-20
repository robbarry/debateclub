from typing import List, Tuple, Optional
from enum import Enum
from pydantic import BaseModel, Field


class Position(str, Enum):
    PRO = "pro"
    CON = "con"


class DebateModel(str, Enum):
    ANTHROPIC = "claude-3-5-sonnet-20241022"
    OPENAI = "gpt-4o"
    GEMINI = "gemini-exp-1206"


class DebateTopic(BaseModel):
    topic: str
    context: str
    pro_position: str
    con_position: str


class ReasoningItem(BaseModel):
    premise: str
    reasoning: str


class DebateArgument(BaseModel):
    position: Position
    introduction: str
    reasoning: List[ReasoningItem]
    rebuttal: str


class JudgmentExtraction(BaseModel):
    pro_claims: List[Tuple[str, str]] = Field(
        ...,
        json_schema_extra={
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["claim", "reasoning"],
            }
        },
    )
    con_claims: List[Tuple[str, str]] = Field(
        ...,
        json_schema_extra={
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["claim", "reasoning"],
            }
        },
    )
    pro_rebuttals: List[str]
    con_rebuttals: List[str]
    logical_fallacies: List[str]


class JudgmentScore(BaseModel):
    logic_score: int  # 1-10
    evidence_score: int  # 1-10
    rebuttal_score: int  # 1-10
    overall_score: int  # 1-10
    reasoning: str


class JudgmentResult(BaseModel):
    winner: Optional[Position]
    pro_score: JudgmentScore
    con_score: JudgmentScore
    explanation: str


class TextResponse(BaseModel):
    text: str
