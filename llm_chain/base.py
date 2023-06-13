"""Contains base classes."""
from abc import ABC, abstractmethod
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field


class Record(BaseModel):
    """TODO."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = Field(
        default_factory=lambda: datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S'),
    )
    metadata: dict = {}

    def __str__(self) -> str:
        return f"timestamp: {self.timestamp}; metadata: {self.metadata}"


class UsageRecord(Record):
    """TODO."""

    total_tokens: int | None = None
    cost: float | None = None

    def __str__(self) -> str:
        return f"timestamp: {self.timestamp}; cost: ${self.cost:.6f}; " \
            f"total_tokens: {self.total_tokens:,}; metadata: {self.metadata}"


class MessageRecord(UsageRecord):
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

    def __str__(self) -> str:
        return f"timestamp: {self.timestamp}; prompt: \"{self.prompt.strip()[0:20]}...\"; "\
            f"response: \"{self.response.strip()[0:20]}...\";  " \
            f"cost: ${self.cost:.6f}; total_tokens: {self.total_tokens:,}; " \
            f"metadata: {self.metadata}"


class EmbeddingsRecord(UsageRecord):
    """TODO."""


class Document(BaseModel):
    """TODO."""

    content: str
    metadata: dict | None


class HistoricalData(ABC):
    """An object that tracks history i.e. `Record` objects."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """TODO."""


class HistoricalUsageRecords(HistoricalData):
    """
    An object that tracks usage history i.e. `UsageRecord` objects (e.g. usage/tokens/costs in chat
    or embeddings model).
    """

    @property
    @abstractmethod
    def history(self) -> list[UsageRecord]:
        """TODO."""

    @property
    def total_tokens(self) -> str:
        """TODO."""
        if self.history and self.history[0].total_tokens is not None:
            return sum(x.total_tokens for x in self.history)
        return None

    @property
    def total_cost(self) -> str:
        """TODO."""
        if self.history and self.history[0].cost is not None:
            return sum(x.cost for x in self.history)
        return None


class LargeLanguageModel(HistoricalUsageRecords):
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
    def history(self) -> list[Record]:
        """TODO."""


class EmbeddingsModel(LargeLanguageModel):
    """TODO."""

    def __init__(self) -> None:
        super().__init__()
        self._history: list[EmbeddingsRecord] = []

    @abstractmethod
    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingsRecord]:
        """TODO."""

    def __call__(self, docs: list[Document]) -> list[list[float]]:
        """TODO."""
        embeddings, metadata = self._run(docs=docs)
        self._history.append(metadata)
        return embeddings

    @property
    def history(self) -> list[EmbeddingsRecord]:
        """TODO."""
        return self._history


class ChatModel(LargeLanguageModel):
    """
    A Model (e.g. ChatGPT-3) is a class that is callable and invoked with a string and returns a
    string. It has helper methods that track the history/usage of the how an instantiated model
    has been used (e.g. processing time, tokens used, or costs incurred; although not all models
    have direct costs like ChatGPT).
    """

    def __init__(self):
        super().__init__()
        self._history: list[MessageRecord] = []

    @abstractmethod
    def _run(self, prompt: str) -> MessageRecord:
        """Subclasses should override this function and generate responses from the LLM."""


    def __call__(self, prompt: str) -> str:
        """
        When the object is called it takes a prompt (string) and returns a response (string).

        Args:
            prompt: the string prompt/question to the model.
        """
        response = self._run(prompt)
        self._history.append(response)
        return response.response

    @property
    def history(self) -> list[MessageRecord]:
        """TODO."""
        return self._history

    @property
    def previous_message(self) -> MessageRecord:
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


class MemoryBuffer(ABC):
    """TODO."""

    @abstractmethod
    def __call__(self, history: list[MessageRecord]) -> list[MessageRecord]:
        """TODO."""


class PromptTemplate(HistoricalUsageRecords):
    """
    A prompt_template is a callable object that takes a prompt (e.g. user query) as input and
    returns a modified prompt. Each prompt_template is given the information it needs when it is
    instantiated. So for example, if a template's job is to search for relevant documents, it's
    provided the vector database when the object is created (not via __call__).
    """

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """TODO."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """TODO."""


class DocumentIndex(HistoricalUsageRecords):
    """
    A DocumentIndex is simply a way of adding and searching for `Document` objects. For example, it
    could be a wrapper around chromadb.

    A DocumentIndex should propagate any total_tokens or total_cost used by the underlying models
    (e.g. if it uses an EmbeddingModel), or return None if not applicable.
    """

    def __call__(
            self,
            value: Document | list[Document],
            n_results: int = 3) -> list[Document] | None:
        """TODO."""
        if isinstance(value, list):
            return self.add(docs=value)
        assert isinstance(value, Document)
        return self.search(doc=value, n_results=n_results)

    @abstractmethod
    def add(self, docs: list[Document]) -> None:
        """Add documents to the underlying index/database."""

    @abstractmethod
    def search(self, doc: Document, n_results: int = 3) -> list[Document]:
        """Search for documents in the underlying index/database."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """TODO."""
