import pytest
from typing import List, Dict, Tuple
from pydantic import ValidationError
from debateclub.arena import (
    DebateArgument,
    DebateArena,
    DebateTopic,
    Position,
    DebateModel,
)
from unittest.mock import MagicMock
from openai import OpenAI
import instructor
from enum import Enum


class Position(str, Enum):
    PRO = "pro"
    CON = "con"


@pytest.fixture
def mock_openai_client():
    mock_client = MagicMock(spec=OpenAI())
    mock_client.chat = MagicMock()
    mock_client.chat.completions = MagicMock()
    return mock_client


@pytest.fixture
def arena(mock_openai_client):
    return DebateArena(gemini_client=None)


@pytest.fixture
def test_topic():
    return DebateTopic(
        topic="Test Topic",
        context="Test Context",
        pro_position="Pro position",
        con_position="Con position",
    )


def test_debate_argument_validation():
    # Valid DebateArgument
    valid_data = {
        "position": "pro",
        "introduction": "This is a valid introduction",
        "reasoning": [
            {"premise": "Premise 1", "reasoning": "Reasoning 1"},
            {"premise": "Premise 2", "reasoning": "Reasoning 2"},
        ],
        "rebuttal": "This is a valid rebuttal",
    }
    try:
        DebateArgument(**valid_data)
    except ValidationError as e:
        pytest.fail(f"Validation error occurred with valid data: {e}")

    # Invalid DebateArgument
    invalid_data = {
        "position": "pro",
        "introduction": "This is an invalid introduction",
        "reasoning": [
            ["Premise 1", "Reasoning 1"],  # This should be a list of dicts now
            ["Premise 2", "Reasoning 2"],
        ],
        "rebuttal": "This is an invalid rebuttal",
    }
    with pytest.raises(ValidationError) as exc_info:
        DebateArgument(**invalid_data)

    assert "reasoning" in str(exc_info.value)


def test_generate_argument_with_openai(mock_openai_client, arena, test_topic):
    # Mock the _create_completion method
    mock_openai_client.chat.completions.create.return_value = MagicMock(
        model_dump_json=lambda: """
        {
            "position": "pro",
            "introduction": "Introduction",
            "reasoning": [
                {"premise": "Premise 1", "reasoning": "Reasoning 1"},
                {"premise": "Premise 2", "reasoning": "Reasoning 2"},
                {"premise": "Premise 3", "reasoning": "Reasoning 3"}
            ],
            "rebuttal": "Rebuttal"
        }
    """
    )

    argument = arena.generate_argument(
        model=(mock_openai_client, "gpt-4o"),
        topic=test_topic,
        position=Position.PRO,
    )

    assert isinstance(argument, DebateArgument)
    assert argument.position == Position.PRO
    assert isinstance(argument.reasoning, list)
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in argument.reasoning
    )

    # check all the values came through correctly
    assert argument.introduction == "Introduction"
    assert argument.reasoning[0] == ("Premise 1", "Reasoning 1")
    assert argument.reasoning[1] == ("Premise 2", "Reasoning 2")
    assert argument.reasoning[2] == ("Premise 3", "Reasoning 3")
    assert argument.rebuttal == "Rebuttal"


def test_generate_argument_with_retries(mock_openai_client, arena, test_topic):
    mock_openai_client.chat.completions.create.side_effect = [
        MagicMock(
            model_dump_json=lambda: """
        {
            "position": "pro",
            "introduction": "Introduction",
            "reasoning": [
                ["Premise 1", "Reasoning 1"], 
                ["Premise 2", "Reasoning 2"],
            ],
            "rebuttal": "Rebuttal"
        }
    """
        ),  # invalid response for first retry
        MagicMock(
            model_dump_json=lambda: """
        {
            "position": "pro",
            "introduction": "Introduction",
            "reasoning": [
                {"premise": "Premise 1", "reasoning": "Reasoning 1"},
                {"premise": "Premise 2", "reasoning": "Reasoning 2"},
                {"premise": "Premise 3", "reasoning": "Reasoning 3"}
            ],
            "rebuttal": "Rebuttal"
        }
    """
        ),  # valid response for second retry
    ]

    argument = arena.generate_argument(
        model=(mock_openai_client, "gpt-4o"),
        topic=test_topic,
        position=Position.PRO,
    )

    assert isinstance(argument, DebateArgument)
    assert argument.position == Position.PRO
    assert isinstance(argument.reasoning, list)
    assert all(
        isinstance(item, tuple) and len(item) == 2 for item in argument.reasoning
    )

    # check all the values came through correctly
    assert argument.introduction == "Introduction"
    assert argument.reasoning[0] == ("Premise 1", "Reasoning 1")
    assert argument.reasoning[1] == ("Premise 2", "Reasoning 2")
    assert argument.reasoning[2] == ("Premise 3", "Reasoning 3")
    assert argument.rebuttal == "Rebuttal"

    # Check retry was used (we expect this to be called twice)
    assert mock_openai_client.chat.completions.create.call_count == 2
