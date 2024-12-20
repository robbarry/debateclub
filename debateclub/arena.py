import os
from typing import List, Tuple, Optional, Union, Any, Dict
from enum import Enum
import instructor
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai
from pydantic import BaseModel, Field
import re
import time
import sqlite3
import json
from math import pow
from prettytable import PrettyTable
from pydantic import ValidationError
import random

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
    introduction: str
    reasoning: List[Dict[str, str]] = Field(
        ...,
        json_schema_extra={
            "items": {
                "type": "object",
                "properties": {
                    "premise": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["premise", "reasoning"],
            }
        },
    )
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


class DebateArena:
    def __init__(self, db_path="debate_arena.db", gemini_client=None):
        self.models = {
            DebateModel.ANTHROPIC: (anthropic_client, "claude-3-5-sonnet-20241022"),
            DebateModel.OPENAI: (openai_client, "gpt-4o"),
            DebateModel.GEMINI: (gemini_client, "gemini-exp-1206"),
        }
        self.round_count = 3  # Standard number of debate rounds
        self.db_path = db_path
        self.gemini_client = gemini_client
        self._init_db()

    def _init_db(self):
        """Initialize the SQLite database and tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS debates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                topic_summary TEXT NOT NULL,
                context TEXT NOT NULL,
                pro_arguments TEXT NOT NULL,
                con_arguments TEXT NOT NULL,
                pro_scores TEXT NOT NULL,
                con_scores TEXT NOT NULL,
                winner TEXT,
                judge_scores TEXT
            )
            """
        )

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS topics (
                topic TEXT PRIMARY KEY,
                summary TEXT NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS elo_ratings (
                model TEXT PRIMARY KEY,
                rating INTEGER NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS wins_losses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                topic TEXT NOT NULL,
                model1 TEXT NOT NULL,
                model2 TEXT NOT NULL,
                winner TEXT,
                model1_elo INTEGER NOT NULL,
                model2_elo INTEGER NOT NULL
            )
            """
        )
        # Initialize elo ratings if not in DB
        for model in DebateModel:
            cursor.execute(
                "INSERT OR IGNORE INTO elo_ratings (model, rating) VALUES (?, ?)",
                (model.value, 1000),
            )

        conn.commit()
        conn.close()

    def _create_completion(
        self,
        client: Union[OpenAI, Anthropic, Any],
        model: str,
        messages: List[dict],
        response_model: type,
        is_json=True,
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
            if is_json:
                return response_model.model_validate_json(text)
            else:
                return response_model(text=text)
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
                if is_json:
                    return response_model.model_validate_json(text)
                else:
                    return response_model(text=text)
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
            if is_json:
                return response_model.model_validate_json(text)
            else:
                return response_model(text=text)

    def generate_topic(
        self,
        judge_model: Tuple[Union[OpenAI, Anthropic, Any], str],
        previous_topics: List[str] = [],
    ) -> DebateTopic:
        client, model = judge_model
        prompt = f"""Generate a balanced, thoughtful debate topic that has clear pro and con positions.
        The topic should be nuanced enough for meaningful discussion but not overly controversial.
        
        Ensure that the debate topic is different from the following previous topics:
         {', '.join(previous_topics)}

        Provide output in the following format:
        {{
            "topic": "The debate topic as a question",
            "context": "Brief background context for the debate",
            "pro_position": "Clear statement of the pro position",
            "con_position": "Clear statement of the con position"
        }}
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
        max_retries=3,
    ) -> DebateArgument:
        client, model_name = model
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
          Reasoning: {[premise for premise, _ in last_argument.reasoning]}
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
                response = self._create_completion(
                    client,
                    model_name,
                    [{"role": "user", "content": context}],
                    DebateArgument,
                    is_json=True,
                )
                response.position = (
                    position  # overwrite the LLM response as it could be wrong still
                )
                # transform dict to tuple.
                response.reasoning = [
                    (item["premise"], item["reasoning"]) for item in response.reasoning
                ]
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

        extraction = self._create_completion(
            client,
            model,
            [{"role": "user", "content": debate_summary}],
            JudgmentExtraction,
        )
        # transform dict to tuple.
        extraction.pro_claims = [
            (item["claim"], item["reasoning"]) for item in extraction.pro_claims
        ]
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

        response = self._create_completion(
            client, model, [{"role": "user", "content": scoring_prompt}], JudgmentResult
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

    def summarize_topic(
        self, judge_model: Tuple[Union[OpenAI, Anthropic, Any], str], topic: DebateTopic
    ) -> str:
        client, model = judge_model
        prompt = f"""Please provide a short, high-level summary of the following debate topic.

          Topic: {topic.topic}
          Context: {topic.context}
          Pro Position: {topic.pro_position}
          Con Position: {topic.con_position}
          """
        response = self._create_completion(
            client,
            model,
            [{"role": "user", "content": prompt}],
            TextResponse,
            is_json=False,
        )
        return response.text

    def _format_arguments(self, arguments: List[DebateArgument]) -> str:
        formatted_args = []
        for i, arg in enumerate(arguments):
            formatted_arg = f"Round {i + 1}:\n"
            formatted_arg += f"  Introduction: {arg.introduction}\n"
            formatted_arg += "  Reasoning:\n"
            for j, (premise, reasoning) in enumerate(arg.reasoning):
                formatted_arg += f"    {j+1}. {premise} (Reasoning: {reasoning})\n"
            formatted_arg += f"  Rebuttal: {arg.rebuttal}\n"
            formatted_args.append(formatted_arg)
        return "\n".join(formatted_args)

    def run_debate(
        self,
    ) -> Tuple[JudgmentResult, JudgmentResult, JudgmentResult, DebateTopic]:
        # Select model types first
        model_types = random.sample(list(DebateModel), 2)
        judge_type = next(m for m in DebateModel if m not in model_types)

        # Get the actual client-model tuples
        debater_models = [(self.models[model_type]) for model_type in model_types]
        judge_model = self.models[judge_type]

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Retrieve previous topics from the database
        cursor.execute("SELECT topic FROM topics")
        previous_topics = [row[0] for row in cursor.fetchall()]

        print("Starting debate...")
        print(f"Debater 1: {model_types[0]}")
        print(f"Debater 2: {model_types[1]}")
        print(f"Judge: {judge_type}")

        # Generate topic using the judge model
        topic = self.generate_topic(judge_model, previous_topics)
        print(f"\nDebate Topic: {topic.topic}")
        print(f"Context: {topic.context}")

        # Generate topic summary for DB storage
        topic_summary = self.summarize_topic(judge_model, topic)

        # Store the topic and summary
        cursor.execute(
            "INSERT OR IGNORE INTO topics (topic, summary) VALUES (?,?)",
            (topic.topic, topic_summary),
        )
        conn.commit()

        # Randomly assign positions
        positions = random.sample([Position.PRO, Position.CON], 2)
        debaters = list(zip(debater_models, positions))

        print(f"\n{debaters[0][0]}: {debaters[0][1].value}")
        print(f"{debaters[1][0]}: {debaters[1][1].value}\n")

        # Run debate rounds
        pro_arguments = []
        con_arguments = []
        last_pro_argument = None
        last_con_argument = None

        for round_num in range(self.round_count):
            print(f"\nRound {round_num + 1}:")
            # Alternate who goes first each round
            if round_num % 2 == 0:
                order = debaters
            else:
                order = debaters[::-1]

            for model, position in order:
                print(f"\n{model[0]} ({position.value}) is presenting...")
                if position == Position.PRO:
                    argument = self.generate_argument(
                        model,
                        topic,
                        position,
                        [last_con_argument] if last_con_argument else None,
                    )
                    last_pro_argument = argument
                    pro_arguments.append(argument)
                else:
                    argument = self.generate_argument(
                        model,
                        topic,
                        position,
                        [last_pro_argument] if last_pro_argument else None,
                    )
                    last_con_argument = argument
                    con_arguments.append(argument)

                print(f"Argument: \n{argument.introduction}")
                print("Reasoning:")
                for i, (premise, reasoning) in enumerate(argument.reasoning):
                    print(f"   {i+1}. {premise} (Reasoning: {reasoning})")
                print(f"Rebuttal: {argument.rebuttal}\n")
        # Get judgments from all three models
        print("\nJudges are evaluating the debate...")
        judgments = []
        for model_type in DebateModel:
            print(f"\n{model_type} is judging...")
            judgment = self.judge_debate(
                self.models[model_type], topic, pro_arguments, con_arguments
            )
            judgments.append(judgment)

        conn.close()

        return tuple(judgments), topic

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

    def record_debate(
        self,
        topic: DebateTopic,
        topic_summary: str,
        pro_arguments: List[DebateArgument],
        con_arguments: List[DebateArgument],
        judgments: Tuple[JudgmentResult, JudgmentResult, JudgmentResult],
        winner: Optional[Position],
        model_types: List[DebateModel],
    ):
        """Record the debate details in the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        pro_scores_json = json.dumps([j.pro_score.model_dump() for j in judgments])
        con_scores_json = json.dumps([j.con_score.model_dump() for j in judgments])
        judge_scores_json = json.dumps([j.model_dump() for j in judgments])

        cursor.execute(
            """
            INSERT INTO debates (
                topic, topic_summary, context, pro_arguments, con_arguments, pro_scores, con_scores, winner, judge_scores
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                topic.topic,
                topic_summary,
                topic.context,
                json.dumps([arg.model_dump() for arg in pro_arguments]),
                json.dumps([arg.model_dump() for arg in con_arguments]),
                pro_scores_json,
                con_scores_json,
                winner.value if winner else None,
                judge_scores_json,
            ),
        )

        # get elos before calculation
        cursor.execute(
            "SELECT rating from elo_ratings WHERE model = ?", (model_types[0].value,)
        )
        model1_elo = cursor.fetchone()[0]
        cursor.execute(
            "SELECT rating from elo_ratings WHERE model = ?", (model_types[1].value,)
        )
        model2_elo = cursor.fetchone()[0]

        # Record to wins/losses table
        cursor.execute(
            """
            INSERT INTO wins_losses (
                topic, model1, model2, winner, model1_elo, model2_elo
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                topic.topic,
                model_types[0].value,
                model_types[1].value,
                winner.value if winner else None,
                model1_elo,
                model2_elo,
            ),
        )

        conn.commit()
        conn.close()

    def calculate_elo(
        self, winner: Optional[DebateModel], loser: Optional[DebateModel]
    ) -> None:
        """Calculates and updates ELO rating"""
        if not winner or not loser:
            return  # do not update elo on a tie

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT rating from elo_ratings WHERE model = ?", (winner.value,)
        )
        winner_elo = cursor.fetchone()[0]
        cursor.execute("SELECT rating from elo_ratings WHERE model = ?", (loser.value,))
        loser_elo = cursor.fetchone()[0]

        k = 32
        expected_winner = 1 / (1 + pow(10, (loser_elo - winner_elo) / 400))
        expected_loser = 1 / (1 + pow(10, (winner_elo - loser_elo) / 400))

        new_winner_elo = int(winner_elo + k * (1 - expected_winner))
        new_loser_elo = int(loser_elo + k * (0 - expected_loser))

        cursor.execute(
            "UPDATE elo_ratings SET rating = ? WHERE model = ?",
            (new_winner_elo, winner.value),
        )
        cursor.execute(
            "UPDATE elo_ratings SET rating = ? WHERE model = ?",
            (new_loser_elo, loser.value),
        )

        conn.commit()
        conn.close()

    def display_win_loss_table(self):
        """Display win/loss table from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute(
            "SELECT topic, model1, model2, winner, model1_elo, model2_elo FROM wins_losses"
        )
        rows = cursor.fetchall()

        table = PrettyTable()
        table.field_names = [
            "Topic",
            "Model 1",
            "Model 2",
            "Winner",
            "Model 1 ELO",
            "Model 2 ELO",
        ]
        for row in rows:
            table.add_row(row)
        print("\nWin/Loss Table:")
        print(table)
        conn.close()


def run_debates(num_runs=1):
    """Run multiple debate simulations"""
    arena = DebateArena(gemini_client=gemini_client)
    try:
        for i in range(num_runs):
            print(f"\n----- Debate Run {i+1} -----")
            judgments, topic = arena.run_debate()
            winner = arena.determine_winner(judgments)

            # Get a summary from the LLM and save it to the DB
            conn = sqlite3.connect(arena.db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT summary FROM topics WHERE topic = ?", (topic.topic,))
            topic_summary = cursor.fetchone()[0]

            # get models
            model_types = random.sample(list(DebateModel), 2)

            arena.record_debate(
                topic,
                topic_summary,
                [
                    a
                    for a in judgments[0].model_dump()["pro_score"].values()
                    if type(a) is list
                ],
                [
                    a
                    for a in judgments[0].model_dump()["con_score"].values()
                    if type(a) is list
                ],
                judgments,
                winner,
                model_types,
            )

            if winner == Position.PRO:
                arena.calculate_elo(model_types[0], model_types[1])
            elif winner == Position.CON:
                arena.calculate_elo(model_types[1], model_types[0])
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

            # Display win/loss table
            arena.display_win_loss_table()

    finally:
        del arena.gemini_client

    # Show the ELO rating table after the simulation is complete
    conn = sqlite3.connect(arena.db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM elo_ratings")
    print("\nELO Ratings:")
    for row in cursor.fetchall():
        print(f"Model: {row[0]}, Rating: {row[1]}")
    conn.close()


if __name__ == "__main__":
    run_debates(num_runs=2)  # Run 2 debates for now