"""Contains base classes."""
from abc import ABC, abstractmethod
import inspect
from typing import Any
from collections.abc import Callable
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field


class Record(BaseModel):
    """An object typically used to track the history of a task/link."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    )
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"timestamp: {self.timestamp}; metadata: {self.metadata}"


class StreamingEvent(Record):
    """An object that contains the information from a streaming event."""

    response: str


class UsageRecord(Record):
    """Represents a record associated with token usage and/or costs."""

    total_tokens: int | None = None
    cost: float | None = None

    def __str__(self) -> str:
        return f"timestamp: {self.timestamp}; cost: ${self.cost or 0:.6f}; " \
            f"total_tokens: {self.total_tokens or 0:,}; metadata: {self.metadata}"


class MessageRecord(UsageRecord):
    """
    A MessageMetaData is a single interaction with an LLM (i.e. a prompt and a response). It's used
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
            f"cost: ${self.cost or 0:.6f}; total_tokens: {self.total_tokens or 0:,}; " \
            f"metadata: {self.metadata}"


class EmbeddingsRecord(UsageRecord):
    """Record associated with an Embeddings request."""


class Document(BaseModel):
    """
    A document consist of content/text and metadata. It can represent anything from file, web-page,
    or subset/chunk of a whole document.
    """

    content: str
    metadata: dict | None


class HistoricalData(ABC):
    """An object that tracks history i.e. `Record` objects."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """A list of Records for tracking events (e.g. messages, requests, searches, etc.)."""


class HistoricalUsageRecords(HistoricalData):
    """
    An object that tracks usage history i.e. `UsageRecord` objects (e.g. usage/tokens/costs in chat
    or embeddings model).
    """

    @property
    @abstractmethod
    def history(self) -> list[UsageRecord]:
        """A list of Records for tracking events (e.g. messages, requests, searches, etc.)."""

    @property
    def total_tokens(self) -> int | None:
        """The total number of tokens associated with the event."""
        if self.history and self.history[0].total_tokens is not None:
            return sum(x.total_tokens for x in self.history)
        return None

    @property
    def cost(self) -> float | None:
        """The total cost associated with the event."""
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
        """Executes the chat request based on the value (e.g. message(s)) passed in."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """A list of Records for tracking chat messages (e.g. prompt/response)."""


class EmbeddingsModel(LargeLanguageModel):
    """A model that produces embeddings for a given piece of text."""

    def __init__(self) -> None:
        super().__init__()
        self._history: list[EmbeddingsRecord] = []

    @abstractmethod
    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingsRecord]:
        """Execute the embeddings request."""

    def __call__(self, docs: list[Document] | list[str] | Document | str) -> list[list[float]]:
        """
        Executes the embeddings request based on the document(s).

        Args:
            docs:
                Either a list of Documents, single Document, or str. Returns the embeddings that
                correspond to the doc(s).
        """
        if not docs:
            return []
        if isinstance(docs, list):
            if isinstance(docs[0], str):
                docs = [Document(content=x) for x in docs]
            else:
                assert isinstance(docs[0], Document)
        elif isinstance(docs, Document):
            docs = [docs]
        elif isinstance(docs, str):
            docs = [Document(content=docs)]
        else:
            raise TypeError("Invalid type.")

        embeddings, metadata = self._run(docs=docs)
        self._history.append(metadata)
        return embeddings

    @property
    def history(self) -> list[EmbeddingsRecord]:
        """A list of EmbeddingsRecord that correspond to each embeddings request."""
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
        Executes the chat request based on the the prompt (string) and returns a response (string).

        Args:
            prompt: the string prompt/question to the model.
        """
        response = self._run(prompt)
        self._history.append(response)
        return response.response

    @property
    def history(self) -> list[MessageRecord]:
        """A list of MessageRecord for tracking chat messages (prompt/response)."""
        return self._history

    @property
    def previous_message(self) -> MessageRecord | None:
        """Returns the last/previous message (MessageMetaData) associated with the chat model."""
        if len(self.history) == 0:
            return None
        return self.history[-1]

    @property
    def previous_prompt(self) -> str | None:
        """Returns the last/previous prompt used in chat model."""
        previous_message = self.previous_message
        if previous_message:
            return previous_message.prompt
        return None

    @property
    def previous_response(self) -> str | None:
        """Returns the last/previous response used in chat model."""
        previous_message = self.previous_message
        if previous_message:
            return previous_message.response
        return None

    @property
    def prompt_tokens(self) -> int | None:
        """
        Returns the total number of prompt_tokens used by the model during this object's lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        # if there is no token_counter then there won't be a way to calculate the number of tokens
        if self.previous_message and self.previous_message.prompt_tokens:
            return sum(x.prompt_tokens for x in self.history)
        return None

    @property
    def response_tokens(self) -> int | None:
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
    """
    Class that has logic to handle the memory (i.e. total context) of the messages sent to an
    LLM.
    """

    @abstractmethod
    def __call__(self, history: list[MessageRecord]) -> list[MessageRecord]:
        """
        Takes the hisitory of messages and returns a modified/reduced list of messages based on the
        memory strategy.
        """


class PromptTemplate(HistoricalUsageRecords):
    """
    A prompt_template is a callable object that takes a prompt (e.g. user query) as input and
    returns a modified prompt. Each prompt_template is given the information it needs when it is
    instantiated. So for example, if a template's job is to search for relevant documents, it's
    provided the vector database when the object is created (not via __call__).
    """

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """Takes user inuput 
         and returns a prompt to the LLM."""

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

    TODO: n_results can be passed during object initialization or when called/searched. The latter
    takes priority.
    """

    def __init__(self, n_results: int = 3) -> None:
        super().__init__()
        self._n_results = n_results

    def __call__(
            self,
            value: Document | str | list[Document],
            n_results: int | None = None) -> list[Document] | None:
        """TODO."""
        if isinstance(value, list):
            return self.add(docs=value)
        if isinstance(value, Document | str):
            return self.search(value=value, n_results=n_results)
        raise TypeError("Invalid Type")

    @abstractmethod
    def add(self, docs: list[Document]) -> None:
        """Add documents to the underlying index/database."""

    @abstractmethod
    def _search(self, doc: Document, n_results: int) -> list[Document]:
        """Search for documents in the underlying index/database."""

    def search(
            self,
            value: Document | str,
            n_results: int | None = None) -> list[Document]:
        """
        Search for documents in the underlying index/database.

        TODO: n_results can be passed during object initialization or when called/searched.
        The latter takes priority.
        """
        if isinstance(value, str):
            value = Document(content=value)
        return self._search(doc=value, n_results=n_results or self._n_results)

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """TODO."""


class Value:
    """TODO."""

    def __init__(self):
        self.value = None

    def __call__(self, value: object | None = None) -> object:
        """TODO."""
        if value:
            self.value = value
        return self.value


class LinkAggregator(ABC):
    """TODO."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """TODO."""

    @property
    def usage_history(self) -> list[UsageRecord]:
        """TODO."""
        return [x for x in self.history if isinstance(x, UsageRecord)]

    @property
    def message_history(self) -> list[MessageRecord]:
        """TODO."""
        return [x for x in self.history if isinstance(x, MessageRecord)]

    @property
    def total_tokens(self) -> int | None:
        """
        Returns the total number of tokens used by the all models during the chain/object's
        lifetime.

        Returns `None` if none of the models knows how to count tokens.
        """
        records = self.usage_history
        totals = [x.total_tokens for x in records if x.total_tokens]
        if not totals:
            return None
        return sum(totals)

    @property
    def cost(self) -> float | None:
        """
        Returns the total cost used by the all models during the chain/object's
        lifetime.

        Returns `None` if none of the models knows how to count cost.
        """
        records = self.usage_history
        totals = [x.cost for x in records if x.cost]
        if not totals:
            return None
        return sum(totals)

    @property
    def prompt_tokens(self) -> int | None:
        """
        Returns the total number of prompt tokens used by the all models during the chain/object's
        lifetime.

        Returns `None` if none of the models knows how to count tokens.
        """
        records = self.message_history
        totals = [x.prompt_tokens for x in records if x.prompt_tokens]
        if not totals:
            return None
        return sum(totals)

    @property
    def response_tokens(self) -> int | None:
        """
        Returns the total number of response tokens used by the all models during the
        chain/object's lifetime.

        Returns `None` if none of the models knows how to count tokens.
        """
        records = self.message_history
        totals = [x.response_tokens for x in records if x.response_tokens]
        if not totals:
            return None
        return sum(totals)


class Chain(LinkAggregator):
    """TODO."""

    def __init__(self, links: list[Callable[[Any], Any]]):
        self._links = links

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """TODO."""
        if not self._links:
            return None
        result = self._links[0](*args, **kwargs)
        if len(self._links) > 1:
            for link in self._links[1:]:
                result = link(result)
        return result

    def __getitem__(self, index: int) -> Callable:
        return self._links[index]

    def __len__(self) -> int:
        return len(self._links)

    @property
    def history(self) -> list[Record]:
        """TODO."""
        histories = [link.history for link in self._links if _has_history(link)]
        # Edge-case: if the same model is used multiple times in the same chain (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the chains because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for history in histories:
            for record in history:
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return sorted(unique_records, key=lambda x: x.timestamp)


class Session(LinkAggregator):
    """
    TODO: A session is a way to aggregate the history of chains, calling a session will call
    the last chain added to the session.
    """

    def __init__(self, chains: list[Chain] | None = None):
        self._chains = chains or []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """TODO."""
        if self._chains:
            return self._chains[-1](*args, **kwargs)
        raise ValueError()

    def append(self, chain: Chain) -> None:
        """TODO."""
        self._chains.append(chain)

    def __len__(self) -> int:
        return len(self._chains)

    @property
    def history(self) -> list[Record]:
        """TODO."""
        # for each history in chain, cycle through each link's history and add to the list of
        # records if it hasn't already been added.
        chains = [chain for chain in self._chains if chain.history]
        # Edge-case: if the same model is used multiple times in the same chain or across different
        # links (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the chains because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for chain in chains:
            for record in chain.history:
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return sorted(unique_records, key=lambda x: x.timestamp)


def _has_history(obj: object) -> bool:
    """TODO."""
    return _has_property(obj, property_name='history') and \
        isinstance(obj.history, list) and \
        len(obj.history) > 0 and \
        isinstance(obj.history[0], Record)


def _has_property(obj: object, property_name: str) -> bool:
    if inspect.isfunction(obj):
        return False
    return hasattr(obj, property_name)


class RequestError(Exception):
    """Class that wraps an error when using requests.get()."""

    def __init__(self, status_code: int, reason: str) -> None:
        self.status_code = status_code
        self.reason = reason
