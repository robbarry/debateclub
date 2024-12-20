import sqlite3
from typing import List, Tuple, Optional, Dict, Any
import json
from debateclub.llms import load_all_models


DB_PATH = "debate_arena.db"
models = load_all_models()


def _init_db():
    """Initialize the SQLite database and tables if they don't exist."""
    conn = sqlite3.connect(DB_PATH)
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
    for model in models.values():
        cursor.execute(
            "INSERT OR IGNORE INTO elo_ratings (model, rating) VALUES (?, ?)",
            (model.model_name(), 1000),
        )

    conn.commit()
    conn.close()


def insert_debate_data(
    topic: str,
    topic_summary: str,
    context: str,
    pro_arguments: List[Dict],
    con_arguments: List[Dict],
    pro_scores: List[Dict],
    con_scores: List[Dict],
    winner: Optional[str],
    judge_scores: List[Dict],
):
    """Inserts debate data into the debates table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute(
        """
        INSERT INTO debates (
            topic, topic_summary, context, pro_arguments, con_arguments, pro_scores, con_scores, winner, judge_scores
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            topic,
            topic_summary,
            context,
            json.dumps(pro_arguments),
            json.dumps(con_arguments),
            json.dumps(pro_scores),
            json.dumps(con_scores),
            winner,
            json.dumps(judge_scores),
        ),
    )
    conn.commit()
    conn.close()


def insert_topic_data(topic: str, summary: str):
    """Inserts topic data into the topics table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT OR IGNORE INTO topics (topic, summary) VALUES (?,?)",
        (topic, summary),
    )
    conn.commit()
    conn.close()


def get_previous_topics() -> List[str]:
    """Retrieves previous topics from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT topic FROM topics")
    topics = [row[0] for row in cursor.fetchall()]
    conn.close()
    return topics


def insert_wins_losses(
    topic: str,
    model1: str,
    model2: str,
    winner: Optional[str],
    model1_elo: int,
    model2_elo: int,
):
    """Inserts win/loss data into the wins_losses table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO wins_losses (
            topic, model1, model2, winner, model1_elo, model2_elo
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        (topic, model1, model2, winner, model1_elo, model2_elo),
    )
    conn.commit()
    conn.close()


def get_elo_rating(model: str) -> int:
    """Retrieves the ELO rating for a given model."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT rating FROM elo_ratings WHERE model = ?", (model,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return 1000


def update_elo_rating(model: str, rating: int):
    """Updates the ELO rating for a given model."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("UPDATE elo_ratings SET rating = ? WHERE model = ?", (rating, model))
    conn.commit()
    conn.close()


def get_all_elo_ratings() -> List[Tuple[str, int]]:
    """Retrieves all ELO ratings from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM elo_ratings")
    ratings = cursor.fetchall()
    conn.close()
    return ratings


def get_topic_summary(topic: str) -> str:
    """Retrieves the summary for a given topic."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT summary FROM topics WHERE topic = ?", (topic,))
    result = cursor.fetchone()
    conn.close()
    if result:
        return result[0]
    return ""


def get_win_loss_data() -> List[Tuple[str, str, str, Optional[str], int, int]]:
    """Retrieves all win/loss data from the database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT topic, model1, model2, winner, model1_elo, model2_elo FROM wins_losses"
    )
    rows = cursor.fetchall()
    conn.close()
    return rows


# Initialize the database on import
_init_db()
