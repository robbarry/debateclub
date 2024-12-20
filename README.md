# DebateClub

DebateClub is an experimental framework for evaluating LLM capabilities through structured debates. It pits different language models against each other in formal debates, with other models acting as judges. Each debate is scored on logic, evidence, and rebuttal quality, with ELO ratings tracked over time.

## Setup

1. Install dependencies listed in `pyproject.toml`
2. Ensure you have the required API keys in your environment variables:
   - `ANTHROPIC_API_KEY`

   - `GEMINI_API_KEY`

   - `OPENAI_API_KEY`

## Usage

1. Run `python -m debateclub.arena` to start debates
2. Results and ELO ratings are stored in `debate_arena.db`

The system automatically generates balanced debate topics, manages turns between models, and provides detailed scoring breakdowns for each round. This can be useful for:
* Comparing reasoning capabilities across different LLMs
* Studying how models handle complex argumentation
* Evaluating models' ability to engage with opposing viewpoints
* Analyzing potential biases in different models' approaches
