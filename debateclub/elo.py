from math import pow
from typing import Optional
from debateclub.models import DebateModel
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


def update_elo_ratings(
    winner: Optional[DebateModel], loser: Optional[DebateModel]
) -> None:
    """Updates ELO ratings in the database."""
    if not winner or not loser:
        return  # do not update elo on a tie

    winner_elo = db.get_elo_rating(winner.value)
    loser_elo = db.get_elo_rating(loser.value)

    new_winner_elo, new_loser_elo = calculate_elo_change(winner_elo, loser_elo)

    db.update_elo_rating(winner.value, new_winner_elo)
    db.update_elo_rating(loser.value, new_loser_elo)
