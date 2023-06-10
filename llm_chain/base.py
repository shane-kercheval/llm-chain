"""Contains base classes."""
from abc import ABC, abstractmethod
from pydantic import BaseModel


class MessageMetaData(BaseModel):
    """
    A MessageMetaData is a single interaction with an LLM (i.e. a prompt and a response. It's used
    to capture additional information about that interaction such as the number of tokens used and
    the corresponding costs.

    prompt: could be string representing a message
    """

    prompt: str
    response: str
    metadata: dict | None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class LLM(ABC):
    """
    A Model (e.g. ChatGPT-3) is a class that is callable and invoked with a string and returns a
    string. It has helper methods that track the history/usage of the how an instantiated model
    has been used (e.g. processing time, tokens used, or costs incurred; although not all models
    have direct costs like ChatGPT).
    """

    def __init__(
            self,
            model_type: str,
            ) -> None:
        """
        Init method.

        Args:
            model_type: either 'instruct', 'chat', or 'embedding'
            token_counter:
                if provided, the `prompt_tokens`, `response_tokens`, and `total_tokens` fields will
                be calculated from the prompt/response
            cost_per_token: cost_per_token
                if provided, the `cost` field will be calculated from the `total_tokens` field.
        """
        super().__init__()
        assert model_type in {'instruct', 'chat', 'embedding'}
        self.model_type = model_type
        # self.token_counter = token_counter
        # self.cost_per_token = cost_per_token
        self.history: list[MessageMetaData] = []

    @abstractmethod
    def _run(self, prompt: str) -> MessageMetaData:
        """Subclasses should override this function and generate responses from the LLM."""


    def __call__(self, prompt: str) -> str:
        """
        When the object is called it takes a prompt (string) and returns a response (string).

        Args:
            prompt: the string prompt/question to the model.
        """
        response = self._run(prompt)
        self.history.append(response)
        return response.response

    @property
    def previous_message(self) -> MessageMetaData:
        """Returns the last/previous message (MessageMetaData) associated with the chat model."""
        if len(self.history) == 0:
            return None
        return self.history[-1]

    @property
    def previous_prompt(self) -> str:
        """Returns the last/previous prompt used in chat model."""
        previous_message = self.previous_message
        if previous_message:
            return previous_message.prompt
        return None

    @property
    def previous_response(self) -> str:
        """Returns the last/previous response used in chat model."""
        previous_message = self.previous_message
        if previous_message:
            return previous_message.response
        return None

    @property
    def total_tokens(self) -> str:
        """
        Returns the total number of tokens used by the model during this object's lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        # if there is no token_counter then there won't be a way to calculate the number of tokens
        if self.previous_message and self.previous_message.total_tokens:
            return sum(x.total_tokens for x in self.history)
        return None

    @property
    def total_prompt_tokens(self) -> str:
        """
        Returns the total number of prompt_tokens used by the model during this object's lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        # if there is no token_counter then there won't be a way to calculate the number of tokens
        if self.previous_message and self.previous_message.prompt_tokens:
            return sum(x.prompt_tokens for x in self.history)
        return None

    @property
    def total_response_tokens(self) -> str:
        """
        Returns the total number of response_tokens used by the model during this object's
        lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        # if there is no token_counter then there won't be a way to calculate the number of tokens
        if self.previous_message and self.previous_message.response_tokens:
            return sum(x.response_tokens for x in self.history)
        return None

    @property
    def total_cost(self) -> str:
        """
        Returns the total cost associated with usage of the model during this object's lifetime.

        Returns `None` if the model does not know how to count costs.
        """
        # if there is no cost_per_token then there won't be a way to calculate the the cost
        if self.previous_message and self.previous_message.cost:
            return sum(x.cost for x in self.history)
        return None


class MemoryBuffer(ABC):
    """TBD."""

    @abstractmethod
    def __call__(self, history: list[MessageMetaData]) -> list[MessageMetaData]:
        """TBD."""


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
