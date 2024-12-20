from typing import List, Optional
from enum import Enum
from pydantic import BaseModel, Field


class Position(str, Enum):
    PRO = "pro"
    CON = "con"


class DebateModel(str, Enum):
    ANTHROPIC = "claude-3-5-sonnet-20241022"
    OPENAI = "gpt-4o"
    GEMINI = "gemini-2.0-flash-exp"


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


# Define a Claim model matching the JSON structure
class Claim(BaseModel):
    claim: str
    reasoning: str


class JudgmentExtraction(BaseModel):
    pro_claims: List[Claim]
    con_claims: List[Claim]
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
