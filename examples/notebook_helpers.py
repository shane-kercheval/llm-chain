"""Helper methods for notebooks."""
from IPython.display import display, Markdown
from llm_chain.base import MessageRecord


def usage_string(
        obj: object,
        cost_precision: int = 5,
        include_prompt_response: bool = True) -> str:
    """Returns a friendly string containing cost/usage."""
    value = \
        f"Cost:           ${obj.cost:.{cost_precision}f}\n" + \
        f"Total Tokens:    {obj.total_tokens:,}\n"

    if include_prompt_response:
        value += \
            f"Prompt Tokens:   {obj.prompt_tokens:,}\n" + \
            f"Response Tokens: {obj.response_tokens:,}\n"

    return value


def mprint(value: str) -> None:
    """Print/display the `value` as markdown in a notebook cell."""
    display(Markdown(value))


def messages_string(messages: list[MessageRecord], cost_precision: int = 5) -> str:
    """
    Returns a string containing the formatted messages; can be used with `mprint` to display the
    message history in a notebook cell.
    """
    value = '---\n\n'
    for index, message in enumerate(messages):
        value += f"### Message {index + 1}\n\n"
        value += f"{message.timestamp}; `${message.cost:.{cost_precision}f}`; `{message.total_tokens:,}` Tokens\n"  # noqa
        value += f"#### Prompt ({message.prompt_tokens} tokens):\n"
        value += f"\n{message.prompt}\n"
        value += f"#### Response ({message.response_tokens} tokens):\n"
        value += f"\n{message.response}\n"
        value += '\n---\n\n'
    return value
