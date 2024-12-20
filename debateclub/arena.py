import random
from typing import List, Tuple, Optional
from tabulate import tabulate
import time

from debateclub.models import (
    Position,
    DebateTopic,
    DebateArgument,
    JudgmentResult,
)
from debateclub.llms import LLMModel
from debateclub import db
from debateclub import elo
from debateclub import argumentation
from debateclub import judging
from debateclub.llms.models_manager import get_models


class DebateArena:
    def __init__(self, db_path="debate_arena.db"):
        self.models = get_models()
        self.round_count = 3
        self.db_path = db_path
        self.table_width = 120  # Maximum width for tables

    def display_round(
        self, round_num: int, debater: str, position: Position, argument: DebateArgument
    ):
        """Display a single round's argument in tabulated format"""
        # Create header for the round
        print(f"\nRound {round_num} - {debater} ({position.value})")

        # Format introduction
        intro_table = [[argument.introduction]]
        print(
            tabulate(intro_table, tablefmt="grid", maxcolwidths=[self.table_width - 4])
        )

        # Format reasoning
        reasoning_data = []
        for i, premise in enumerate(argument.reasoning, 1):
            reasoning_data.append([f"Point {i}", premise.premise, premise.reasoning])

        print(
            tabulate(
                reasoning_data,
                headers=["#", "Premise", "Reasoning"],
                tablefmt="grid",
                maxcolwidths=[
                    6,
                    (self.table_width - 10) // 2,
                    (self.table_width - 10) // 2,
                ],
            )
        )

        # Format rebuttal
        rebuttal_table = [[argument.rebuttal]]
        print("\nRebuttal:")
        print(
            tabulate(
                rebuttal_table, tablefmt="grid", maxcolwidths=[self.table_width - 4]
            )
        )

    def display_judgment(self, judge_num: int, judgment: JudgmentResult):
        """Display a single judgment in tabulated format"""
        print(f"\nJudge {judge_num} Decision:")

        # Winner and explanation
        decision_data = [
            ["Winner", judgment.winner.value if judgment.winner else "Tie"],
            ["Explanation", judgment.explanation],
        ]
        print(
            tabulate(
                decision_data, tablefmt="grid", maxcolwidths=[15, self.table_width - 19]
            )
        )

        # Scores table
        scores_data = [
            ["Metric", "Pro Score", "Con Score"],
            ["Logic", judgment.pro_score.logic_score, judgment.con_score.logic_score],
            [
                "Evidence",
                judgment.pro_score.evidence_score,
                judgment.con_score.evidence_score,
            ],
            [
                "Rebuttal",
                judgment.pro_score.rebuttal_score,
                judgment.con_score.rebuttal_score,
            ],
            [
                "Overall",
                judgment.pro_score.overall_score,
                judgment.con_score.overall_score,
            ],
        ]
        print(tabulate(scores_data, headers="firstrow", tablefmt="grid"))

        # Reasoning table
        reasoning_data = [
            ["Side", "Reasoning"],
            ["Pro", judgment.pro_score.reasoning],
            ["Con", judgment.con_score.reasoning],
        ]
        print(
            tabulate(
                reasoning_data,
                headers="firstrow",
                tablefmt="grid",
                maxcolwidths=[10, self.table_width - 14],
            )
        )

    def display_win_loss_table(self):
        """Display win/loss table from database."""
        rows = db.get_win_loss_data()
        print("\nWin/Loss Table:")
        print(
            tabulate(
                rows,
                headers=[
                    "Topic",
                    "Model 1",
                    "Model 2",
                    "Winner",
                    "Model 1 ELO",
                    "Model 2 ELO",
                ],
                tablefmt="grid",
                maxcolwidths=[30, 15, 15, 10, 12, 12],
            )
        )

    def generate_topic(
        self,
        judge_model_name: str,
        previous_topics: List[str] = [],
    ) -> DebateTopic:
        """Generates a debate topic using the specified LLM."""
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

        return judging._create_completion(
            self.models[judge_model_name],
            [{"role": "user", "content": prompt}],
            DebateTopic,
        )

    def run_debate(
        self,
    ) -> Tuple[Tuple[JudgmentResult, JudgmentResult, JudgmentResult], DebateTopic]:
        """Runs a single debate."""
        # Select model types first
        selected_models = random.sample(list(self.models.values()), 2)
        judge_model = next(m for m in self.models.values() if m not in selected_models)
        debater_models = [model for model in selected_models]

        # Get previous topics
        previous_topics = db.get_previous_topics()

        # Display debate setup
        setup_data = [
            ["Role", "Model"],
            ["Debater 1", selected_models[0].model_name()],
            ["Debater 2", selected_models[1].model_name()],
            ["Judge", judge_model.model_name()],
        ]
        print(tabulate(setup_data, headers="firstrow", tablefmt="grid"))

        # Generate topic using the judge model
        topic = self.generate_topic(judge_model.model_name(), previous_topics)

        # Display topic information
        topic_data = [["Topic", topic.topic], ["Context", topic.context]]
        print(
            tabulate(
                topic_data, tablefmt="grid", maxcolwidths=[15, self.table_width - 19]
            )
        )

        # Generate topic summary for DB storage
        topic_summary = judging.summarize_topic(judge_model.model_name(), topic)
        db.insert_topic_data(topic.topic, topic_summary)

        # Randomly assign positions
        positions = random.sample([Position.PRO, Position.CON], 2)
        debaters = list(zip(debater_models, positions))

        positions_data = [
            ["Model", "Position"],
            [debaters[0][0].model_name(), debaters[0][1].value],
            [debaters[1][0].model_name(), debaters[1][1].value],
        ]
        print(tabulate(positions_data, headers="firstrow", tablefmt="grid"))

        # Run debate rounds
        pro_arguments = []
        con_arguments = []
        last_pro_argument = None
        last_con_argument = None

        for round_num in range(self.round_count):
            if round_num % 2 == 0:
                debater_order = debaters
            else:
                debater_order = debaters[::-1]

            for model, position in debater_order:
                argument = argumentation.generate_argument(
                    model.model_name(),
                    topic,
                    position,
                    (
                        [last_con_argument]
                        if last_con_argument and position == Position.PRO
                        else [last_pro_argument] if last_pro_argument else None
                    ),
                )

                if position == Position.PRO:
                    last_pro_argument = argument
                    pro_arguments.append(argument)
                else:
                    last_con_argument = argument
                    con_arguments.append(argument)

                self.display_round(
                    round_num + 1, model.model_name(), position, argument
                )

        # Get judgments
        print("\nJudges are evaluating the debate...")
        judgments = []
        for i, model in enumerate(self.models.values(), 1):
            judgment = judging.judge_debate(
                model.model_name(), topic, pro_arguments, con_arguments
            )
            self.display_judgment(i, judgment)
            judgments.append(judgment)

        return tuple(judgments), topic

    def determine_winner(
        self, judgments: Tuple[JudgmentResult, JudgmentResult, JudgmentResult]
    ) -> Optional[Position]:
        """Determines the winner based on the judgments."""
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
        model_types: List[LLMModel],
    ):
        """Record the debate details in the database"""

        pro_scores_json = [j.pro_score.model_dump() for j in judgments]
        con_scores_json = [j.con_score.model_dump() for j in judgments]
        judge_scores_json = [j.model_dump() for j in judgments]

        db.insert_debate_data(
            topic.topic,
            topic_summary,
            topic.context,
            [arg.model_dump() for arg in pro_arguments],
            [arg.model_dump() for arg in con_arguments],
            pro_scores_json,
            con_scores_json,
            winner.value if winner else None,
            judge_scores_json,
        )

        # get elos before calculation
        model1_elo = db.get_elo_rating(model_types[0].model_name())
        model2_elo = db.get_elo_rating(model_types[1].model_name())

        # Record to wins/losses table
        db.insert_wins_losses(
            topic.topic,
            model_types[0].model_name(),
            model_types[1].model_name(),
            winner.value if winner else None,
            model1_elo,
            model2_elo,
        )


def run_debates(num_runs=1):
    """Run multiple debate simulations"""
    arena = DebateArena()
    try:
        for i in range(num_runs):
            print(f"\n----- Debate Run {i+1} -----")
            judgments, topic = arena.run_debate()
            winner = arena.determine_winner(judgments)

            # Get a summary from the LLM and save it to the DB
            topic_summary = db.get_topic_summary(topic.topic)

            # get models
            selected_models = random.sample(list(arena.models.values()), 2)

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
                selected_models,
            )

            if winner == Position.PRO:
                elo.update_elo_ratings(
                    selected_models[0].model_name(), selected_models[1].model_name()
                )
            elif winner == Position.CON:
                elo.update_elo_ratings(
                    selected_models[1].model_name(), selected_models[0].model_name()
                )
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
        pass  # No need to delete clients here as they are managed within the model modules

    # Show the ELO rating table after the simulation is complete
    elo_ratings = db.get_all_elo_ratings()
    print("\nELO Ratings:")
    for row in elo_ratings:
        print(f"Model: {row[0]}, Rating: {row[1]}")


if __name__ == "__main__":
    run_debates(num_runs=2)  # Run 2 debates for now
