"""TBD."""
from llm_chain.base import MemoryBuffer, MessageMetaData


class MemoryBufferWindow(MemoryBuffer):
    """TBD."""

    def __init__(self, last_n_messages: int) -> None:
        super().__init__()
        self.last_n_messages = last_n_messages

    def __call__(self, history: list[MessageMetaData]) -> list[MessageMetaData]:
        """TBD."""
        if self.last_n_messages == 0:
            return []
        return history[-self.last_n_messages:]


class MemoryBufferTokenWindow(MemoryBuffer):
    """TBD."""

    def __init__(self, last_n_tokens: int) -> None:
        super().__init__()
        self.last_n_tokens = last_n_tokens

    def __call__(self, history: list[MessageMetaData]) -> list[MessageMetaData]:
        """TBD."""
        history = reversed(history)
        memory = []
        tokens_used = 0
        for message in history:
            # if the message's tokens plus the tokens already used in the memory is more than the
            # threshold then we need to break and avoid adding more memory
            if message.total_tokens + tokens_used > self.last_n_tokens:
                break
            memory.append(message)
            tokens_used += message.total_tokens
        return reversed(memory)

