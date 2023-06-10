"""Contains base classes."""
from abc import ABC, abstractmethod
from pydantic import BaseModel


class MessageMetaData(BaseModel):
    """
    A MessageMetaData is a single interaction with an LLM (i.e. a prompt and a response. It's used
    to capture additional information about that interaction such as the number of tokens used and
    the corresponding costs.
    """

    prompt: str
    response: str
    metadata: dict | None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None


class EmbeddingsMetaData(BaseModel):
    """TODO."""

    total_tokens: int
    cost: float | None = None


class Document(BaseModel):
    """TODO."""

    content: str
    metadata: dict | None
    embedding: list | None  # pydantic doesn't seem to support np.ndarray...


class LargeLanguageModel(ABC):
    """
    A Model (e.g. ChatGPT-3, or `text-embedding-ada-002` (embeddings model)) is a class that is
    callable and given some input (e.g. prompt (chat) or documents (embeddings)) and returns a
    response (e.g. string or documents).
    It has helper methods that track the history/usage of the how an instantiated model
    has been used (e.g. processing time, tokens used, or costs incurred; although not all models
    have direct costs like ChatGPT e.g. local models).
    """

    @abstractmethod
    def __call__(self, value: object) -> object:
        """TODO."""


    @property
    @abstractmethod
    def total_tokens(self) -> str:
        """
        Returns the total number of tokens used by the model during this object's lifetime.

        Returns `None` if the model does not know how to count tokens.
        """

    @property
    @abstractmethod
    def total_cost(self) -> str:
        """
        Returns the total cost associated with usage of the model during this object's lifetime.

        Returns `None` if the model does not know how to count costs.
        """


class EmbeddingsModel(LargeLanguageModel):
    """TODO."""

    def __init__(self) -> None:
        super().__init__()
        self.history = []

    @abstractmethod
    def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingsMetaData]:
        """TODO."""

    def __call__(self, docs: list[Document]) -> list[Document]:
        """TODO."""
        docs, metadata = self._run(docs=docs)
        self.history.append(metadata)
        return docs

    @property
    def total_tokens(self) -> str:
        """
        Returns the total number of tokens used by the model during this object's lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        if self.history:
            return sum(x.total_tokens for x in self.history)
        return None

    @property
    def total_cost(self) -> str:
        """
        Returns the total cost associated with usage of the model during this object's lifetime.

        Returns `None` if the model does not know how to count costs.
        """
        if self.history and self.history[0].cost is not None:
            return sum(x.cost for x in self.history)
        return None


class ChatModel(LargeLanguageModel):
    """
    A Model (e.g. ChatGPT-3) is a class that is callable and invoked with a string and returns a
    string. It has helper methods that track the history/usage of the how an instantiated model
    has been used (e.g. processing time, tokens used, or costs incurred; although not all models
    have direct costs like ChatGPT).
    """

    def __init__(self):
        super().__init__()
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
    """TODO."""

    @abstractmethod
    def __call__(self, history: list[MessageMetaData]) -> list[MessageMetaData]:
        """TODO."""

