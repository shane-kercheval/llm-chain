"""
Memory refers to the information sent to a chat model (i.e. how much memory/context the model is
given). Within the context of an OpenAI model, it is the list of messages (a list of dictionaries
where the `role` is `system`, `assistant`, or `user`). Sending the entire list of messages means
the model uses the entire history as context for the answer. However, we can't keep sending the
entire history indefinitely, since we will exceed the maximum context length.

The classes defined below are used to create different strategies for managing memory.

They can be used with, for example, the OpenAIChat model by passing a MemoryBuffer object to the
memory_strategy variable when initializing the model object.
"""
from llm_chain.base import MemoryBuffer, MessageRecord


class MemoryBufferMessageWindow(MemoryBuffer):
    """Returns the last `n` number of messages."""

    def __init__(self, last_n_messages: int) -> None:
        super().__init__()
        self.last_n_messages = last_n_messages

    def __call__(self, history: list[MessageRecord]) -> list[MessageRecord]:
        """
        Takes a list of `MessageRecord` objects and returns the last `n` messages based on the
        `last_n_message` variable set during initialization.
        """
        if self.last_n_messages == 0:
            return []
        return history[-self.last_n_messages:]


class MemoryBufferTokenWindow(MemoryBuffer):
    """Returns the last x number of messages that are within a certain threshold of tokens."""

    def __init__(self, last_n_tokens: int) -> None:
        super().__init__()
        self.last_n_tokens = last_n_tokens

    def __call__(self, history: list[MessageRecord]) -> list[MessageRecord]:
        """
        Takes a list of `MessageRecord` objects and returns the last x messages where the
        aggregated number of tokens is less than the `last_n_message` variable set during
        initialization.
        """
        history = reversed(history)
        memory = []
        tokens_used = 0
        for message in history:
            # if the message's tokens plus the tokens that are already used in the memory is more
            # than the threshold then we need to break and avoid adding more memory
            if message.total_tokens + tokens_used > self.last_n_tokens:
                break
            memory.append(message)
            tokens_used += message.total_tokens
        return reversed(memory)
