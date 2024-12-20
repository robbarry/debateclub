import os
import random
from typing import List, Tuple, Optional, Union, Any
from enum import Enum
import instructor
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from pydantic import BaseModel
import re
import time

# Initialize the clients with instructor
openai_client = instructor.patch(OpenAI())
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_client = genai.GenerativeModel("gemini-exp-1206")  # Handle Gemini directly


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


class DebateArgument(BaseModel):
    position: Position
    argument: str
    key_points: List[str]
    counter_arguments: List[str]


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


class DebateArena:
    def __init__(self):
        self.models = {
            DebateModel.ANTHROPIC: (anthropic_client, "claude-3-5-sonnet-20241022"),
            DebateModel.OPENAI: (openai_client, "gpt-4o"),
            DebateModel.GEMINI: (gemini_client, "gemini-exp-1206"),
        }
        self.round_count = 3  # Standard number of debate rounds

    def _create_completion(
        self,
        client: Union[OpenAI, Anthropic, Any],
        model: str,
        messages: List[dict],
        response_model: type,
    ) -> Any:
        """Handle different client interfaces for completion creation"""
        if isinstance(client, Anthropic):
            # Convert messages to Anthropic format and use messages.create
            content = "\n\n".join(msg["content"] for msg in messages)
            response = client.messages.create(
                model=model,
                max_tokens=1024,
                messages=[{"role": "user", "content": content}],
            )
            # Parse the response to match the response_model
            text = response.content[0].text

            # Remove invalid control characters (e.g., null bytes)
            text = "".join(ch for ch in text if 0x20 <= ord(ch) < 0x10000)
            return response_model.model_validate_json(text)

        elif isinstance(client, genai.GenerativeModel):
            # Handle Gemini's interface
            content = "\n\n".join(msg["content"] for msg in messages)
            response = client.generate_content(content)
            # Parse the response to match the response_model
            if hasattr(response, "text") and response.text:
                text = response.text
                # Remove markdown code blocks if present
                text = re.sub(
                    r"```(?:json)?\s*(.*)\s*```", r"\1", text, flags=re.DOTALL
                )
                # Remove invalid control characters (e.g., null bytes)
                text = "".join(ch for ch in text if 0x20 <= ord(ch) < 0x10000)
                return response_model.model_validate_json(text)
            else:
                raise ValueError(
                    f"Gemini Response did not include a response text: {response}"
                )
        else:
            # Use instructor's patched interface for OpenAI
            response = client.chat.completions.create(
                model=model, response_model=response_model, messages=messages
            )

            text = response.model_dump_json()

            # Remove invalid control characters (e.g., null bytes)
            text = "".join(ch for ch in text if 0x20 <= ord(ch) < 0x10000)
            return response_model.model_validate_json(text)

    def generate_topic(
        self, judge_model: Tuple[Union[OpenAI, Anthropic, Any], str]
    ) -> DebateTopic:
        client, model = judge_model
        prompt = """Generate a balanced, thoughtful debate topic that has clear pro and con positions.
        The topic should be nuanced enough for meaningful discussion but not overly controversial.
        
        Provide output in the following format:
        {
            "topic": "The debate topic as a question",
            "context": "Brief background context for the debate",
            "pro_position": "Clear statement of the pro position",
            "con_position": "Clear statement of the con position"
        }
        """

        return self._create_completion(
            client, model, [{"role": "user", "content": prompt}], DebateTopic
        )

    def generate_argument(
        self,
        model: Tuple[Union[OpenAI, Anthropic, Any], str],
        topic: DebateTopic,
        position: Position,
        previous_arguments: Optional[List[DebateArgument]] = None,
    ) -> DebateArgument:
        client, model_name = model

        context = f"""Topic: {topic.topic}
        Your position: {position.value}
        Context: {topic.context}
        {'Pro position: ' + topic.pro_position if position == Position.PRO else 'Con position: ' + topic.con_position}
        
        Provide a detailed and comprehensive argument in support of your position.
        Your argument should be well-reasoned, and include supporting key points, and anticipated counter arguments to your position.
        
        Provide your response in the following format:
        {{
            "position": "{position.value}",
            "argument": "Your main argument. It should be very comprehensive",
            "key_points": ["Point 1", "Point 2", "Point 3"],
            "counter_arguments": ["Anticipated counter 1", "Anticipated counter 2"]
        }}
        """
        if previous_arguments:
            context += "\nPrevious arguments:\n" + "\n".join(
                f"Round {i+1}: {arg.argument}"
                for i, arg in enumerate(previous_arguments)
            )

        return self._create_completion(
            client, model_name, [{"role": "user", "content": context}], DebateArgument
        )

    def judge_debate(
        self,
        judge_model: Tuple[Union[OpenAI, Anthropic, Any], str],
        topic: DebateTopic,
        pro_arguments: List[DebateArgument],
        con_arguments: List[DebateArgument],
    ) -> JudgmentResult:
        client, model = judge_model

        debate_summary = f"""Topic: {topic.topic}
        Context: {topic.context}
        
        Pro Position Arguments:
        {self._format_arguments(pro_arguments)}
        
        Con Position Arguments:
        {self._format_arguments(con_arguments)}
        
        Judge this debate and provide your judgment in the following format:
        {{
            "winner": "pro or con",
            "pro_score": {{
                "logic_score": 1-10,
                "evidence_score": 1-10,
                "rebuttal_score": 1-10,
                "overall_score": 1-10,
                "reasoning": "Explanation of scores"
            }},
            "con_score": {{
                "logic_score": 1-10,
                "evidence_score": 1-10,
                "rebuttal_score": 1-10,
                "overall_score": 1-10,
                "reasoning": "Explanation of scores"
            }},
            "explanation": "Detailed explanation of your decision"
        }}
        """

        return self._create_completion(
            client, model, [{"role": "user", "content": debate_summary}], JudgmentResult
        )

    def _format_arguments(self, arguments: List[DebateArgument]) -> str:
        return "\n".join(
            f"Round {i+1}: {arg.argument}" for i, arg in enumerate(arguments)
        )

    def run_debate(self) -> Tuple[JudgmentResult, JudgmentResult, JudgmentResult]:
        # Select model types first
        model_types = random.sample(list(DebateModel), 2)
        judge_type = next(m for m in DebateModel if m not in model_types)

        # Get the actual client-model tuples
        debater_models = [(self.models[model_type]) for model_type in model_types]
        judge_model = self.models[judge_type]

        print("Starting debate...")
        print(f"Debater 1: {model_types[0]}")
        print(f"Debater 2: {model_types[1]}")
        print(f"Judge: {judge_type}")

        # Generate topic using the judge model
        topic = self.generate_topic(judge_model)
        print(f"\nDebate Topic: {topic.topic}")
        print(f"Context: {topic.context}")

        # Randomly assign positions
        positions = random.sample([Position.PRO, Position.CON], 2)
        debaters = list(zip(debater_models, positions))

        print(f"\n{debaters[0][0]}: {debaters[0][1].value}")
        print(f"{debaters[1][0]}: {debaters[1][1].value}\n")

        # Run debate rounds
        pro_arguments = []
        con_arguments = []

        for round_num in range(self.round_count):
            print(f"\nRound {round_num + 1}:")
            # Alternate who goes first each round
            if round_num % 2 == 0:
                order = debaters
            else:
                order = debaters[::-1]

            for model, position in order:
                print(f"\n{model[0]} ({position.value}) is presenting...")
                argument = self.generate_argument(
                    model,
                    topic,
                    position,
                    pro_arguments if position == Position.PRO else con_arguments,
                )

                print(f"Argument: {argument.argument}")
                print("Key points:", ", ".join(argument.key_points))

                if position == Position.PRO:
                    pro_arguments.append(argument)
                else:
                    con_arguments.append(argument)

        # Get judgments from all three models
        print("\nJudges are evaluating the debate...")
        judgments = []
        for model_type in DebateModel:
            print(f"\n{model_type} is judging...")
            judgment = self.judge_debate(
                self.models[model_type], topic, pro_arguments, con_arguments
            )
            judgments.append(judgment)

        return tuple(judgments)

    def determine_winner(
        self, judgments: Tuple[JudgmentResult, JudgmentResult, JudgmentResult]
    ) -> Optional[Position]:
        # Count votes for each position
        pro_votes = sum(1 for j in judgments if j.winner == Position.PRO)
        con_votes = sum(1 for j in judgments if j.winner == Position.CON)

        if pro_votes > con_votes:
            return Position.PRO
        elif con_votes > pro_votes:
            return Position.CON
        else:
            # In case of a tie, compare average overall scores
            pro_avg = sum(j.pro_score.overall_score for j in judgments) / len(judgments)
            con_avg = sum(j.con_score.overall_score for j in judgments) / len(judgments)

            if pro_avg > con_avg:
                return Position.PRO
            elif con_avg > pro_avg:
                return Position.CON
            else:
                return None  # True tie


if __name__ == "__main__":
    arena = DebateArena()
    judgments = arena.run_debate()
    winner = arena.determine_winner(judgments)
    # Explicitly delete the Gemini client to encourage faster shutdown
    del gemini_client

    print(f"\nDebate Winner: {winner.value if winner else 'Tie'}")
    for i, judgment in enumerate(judgments):
        print(f"\nJudge {i+1} Decision:")
        print(f"Winner: {judgment.winner.value if judgment.winner else 'Tie'}")
        print(f"Explanation: {judgment.explanation}")
        print("\nPro Scores:")
        print(f"Logic: {judgment.pro_score.logic_score}")
        print(f"Evidence: {judgment.pro_score.evidence_score}")
        print(f"Rebuttal: {judgment.pro_score.rebuttal_score}")
        print(f"Overall: {judgment.pro_score.overall_score}")
        print(f"Reasoning: {judgment.pro_score.reasoning}")
        print("\nCon Scores:")
        print(f"Logic: {judgment.con_score.logic_score}")
        print(f"Evidence: {judgment.con_score.evidence_score}")
        print(f"Rebuttal: {judgment.con_score.rebuttal_score}")
        print(f"Overall: {judgment.con_score.overall_score}")
        print(f"Reasoning: {judgment.con_score.reasoning}")
    time.sleep(0.1)  # small delay to allow the resources to clean up
