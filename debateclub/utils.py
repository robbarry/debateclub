from typing import List
from debateclub.models import DebateArgument


def format_arguments(arguments: List[DebateArgument]) -> str:
    """Formats a list of debate arguments into a readable string."""
    formatted_args = []
    for i, arg in enumerate(arguments):
        formatted_arg = f"Round {i + 1}:\n"
        formatted_arg += f"  Introduction: {arg.introduction}\n"
        formatted_arg += "  Reasoning:\n"
        for j, premise in enumerate(arg.reasoning):
            formatted_arg += (
                f"    {j+1}. {premise.premise} (Reasoning: {premise.reasoning})\n"
            )
        formatted_arg += f"  Rebuttal: {arg.rebuttal}\n"
        formatted_args.append(formatted_arg)
    return "\n".join(formatted_args)
