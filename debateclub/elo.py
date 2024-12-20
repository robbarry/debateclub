from math import pow
from typing import Optional
from debateclub.llms import load_all_models
from debateclub import db


def calculate_elo_change(
    winner_elo: int, loser_elo: int, k: int = 32
) -> tuple[int, int]:
    """Calculates the new ELO ratings for winner and loser."""
    expected_winner = 1 / (1 + pow(10, (loser_elo - winner_elo) / 400))
    expected_loser = 1 / (1 + pow(10, (winner_elo - loser_elo) / 400))

    new_winner_elo = int(winner_elo + k * (1 - expected_winner))
    new_loser_elo = int(loser_elo + k * (0 - expected_loser))

    return new_winner_elo, new_loser_elo


def update_elo_ratings(winner: Optional[str], loser: Optional[str]) -> None:
    """Updates ELO ratings in the database."""
    if not winner or not loser:
        return  # do not update elo on a tie

    winner_elo = db.get_elo_rating(winner)
    loser_elo = db.get_elo_rating(loser)

    new_winner_elo, new_loser_elo = calculate_elo_change(winner_elo, loser_elo)

    db.update_elo_rating(winner, new_winner_elo)
    db.update_elo_rating(loser, new_loser_elo)
