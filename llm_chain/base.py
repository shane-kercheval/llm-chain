"""Contains base classes."""
from abc import ABC, abstractmethod
from typing import Any
from collections.abc import Callable
from uuid import uuid4
from datetime import datetime
from pydantic import BaseModel, Field

from llm_chain.utilities import has_property


class Record(BaseModel):
    """Used to track the history of a task or link."""

    uuid: str = Field(default_factory=lambda: str(uuid4()))
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
    )
    metadata: dict = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"timestamp: {self.timestamp}; metadata: {self.metadata}"


class StreamingEvent(Record):
    """Contains the information from a streaming event."""

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
    A MessageRecord represents a single interaction with an LLM, encompassing a prompt and its
    corresponding response. Its purpose is to record details of the interaction, including the
    token count and associated costs.
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
    A Document comprises both content (text) and metadata, allowing it to represent a wide range of
    entities such as files, web pages, or even specific sections within a larger document.
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
    A LargeLanguageModel, such as ChatGPT-3 or text-embedding-ada-002 (an embeddings model), is a
    class designed to be callable. Given specific inputs, such as prompts for chat-based models or
    documents for embeddings models, it generates meaningful responses, which can be in the form of
    strings or documents.

    Additionally, a LargeLanguageModel is equipped with helpful auxiliary methods that enable
    tracking and analysis of its usage history. These methods provide valuable insights into
    metrics like token consumption and associated costs. It's worth noting that not all models
    incur direct costs, as is the case with ChatGPT; for example, offline models.
    """

    @abstractmethod
    def __call__(self, value: object) -> object:
        """Executes the chat request based on the value (e.g. message(s)) passed in."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """A list of Records for tracking chat messages (e.g. prompt/response)."""


class EmbeddingsModel(LargeLanguageModel):
    """A model that produces embeddings for any given text input."""

    def __init__(self) -> None:
        super().__init__()
        self._history: list[EmbeddingsRecord] = []

    @abstractmethod
    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingsRecord]:
        """Execute the embeddings request."""

    def __call__(self, docs: list[Document] | list[str] | Document | str) -> list[list[float]]:
        """
        Executes the embeddings request based on the document(s) provided.

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
    The ChatModel class represents a callable entity, such as ChatGPT-3, that takes a string as
    input and returns a string. It provides auxiliary methods to monitor the usage history of an
    instantiated model, including metrics like tokens used or costs incurred.
    """

    def __init__(self):
        super().__init__()
        self._history: list[MessageRecord] = []

    @abstractmethod
    def _run(self, prompt: str) -> MessageRecord:
        """Subclasses should override this function and generate responses from the LLM."""


    def __call__(self, prompt: str) -> str:
        """
        Executes a chat request based on the given prompt and returns a response.

        Args:
            prompt: The prompt or question to be sent to the model.
        """
        response = self._run(prompt)
        self._history.append(response)
        return response.response

    @property
    def history(self) -> list[MessageRecord]:
        """A list of MessageRecord objects for tracking chat messages (prompt/response)."""
        return self._history

    @property
    def previous_message(self) -> MessageRecord | None:
        """Returns the last/previous message (MessageRecord) associated with the chat model."""
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
    A PromptTemplate is a callable object that takes a prompt (e.g. user query) as input and
    returns a modified prompt. Each PromptTemplate is provided with the necessary information
    during instantiation. For instance, if a template's purpose is to search for relevant
    documents, it is given the vector database when the object is created, rather than via the
    `__call__` method.
    """

    @abstractmethod
    def __call__(self, prompt: str) -> str:
        """Takes the original prompt (user inuput) and returns a modified prompt."""

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """Propagate any underlying history from e.g. embeddings models."""


class DocumentIndex(HistoricalUsageRecords):
    """
    A `DocumentIndex` is a mechanism for adding and searching for `Document` objects. It can be
    thought of as a wrapper around chromadb or any other similar database.

    A `DocumentIndex` object should propagate any `total_tokens` or `total_cost` used by the
    underlying models, such as an `EmbeddingModel`. If these metrics are not applicable, the
    `DocumentIndex` should return `None`.
    """

    def __init__(self, n_results: int = 3) -> None:
        """
        Args:
            n_results: the number of search-results (from the document index) to return.
        """
        super().__init__()
        self._n_results = n_results

    def __call__(
            self,
            value: Document | str | list[Document],
            n_results: int | None = None) -> list[Document] | None:
        """
        When the object is called, it can either invoke the `add` method (if the `value` passed in
        is a list) or the `search` method (if the `value` passed in is a string or Document). This
        flexible functionality allows the object to be seamlessly integrated into a chain, enabling
        the addition of documents to the index or searching for documents, based on input.

        Args:
            value:
                The value used to determine and retrieve similar Documents.
                Please refer to the description above for more details.
            n_results:
                The maximum number of results to be returned. If provided, it overrides the
                `n_results` parameter specified during initialization (`__init__`).

        Returns:
            If `value` is a list (i.e. the `add` function is called), this method returns None.
            If `value` is a string or Document (i.e the `search` function is called), this method
            returns the search results.
        """
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
        """Search for documents in the underlying index/database based on `doc."""

    def search(
            self,
            value: Document | str,
            n_results: int | None = None) -> list[Document]:
        """
        Search for documents in the underlying index/database.

        Args:
            value:
                The value used to determine and retrieve similar Documents.
            n_results:
                The maximum number of results to be returned. If provided, it overrides the
                `n_results` parameter specified during initialization (`__init__`).
        """
        if isinstance(value, str):
            value = Document(content=value)
        return self._search(doc=value, n_results=n_results or self._n_results)

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """Propagates the history of any underlying models (e.g. embeddings model)."""


class Value:
    """
    The Value class provides a convenient caching mechanism within the chain.
    The `Value` object is callable, allowing it to cache and return values when provided as
    arguments. When called without a value, it retrieves and returns the cached value.
    """

    def __init__(self):
        self.value = None

    def __call__(self, value: object | None = None) -> object:
        """
        When a `value` is provided, it gets cached and returned.
        If no `value` is provided, the previously cached value is returned (or None if no value has
        been cached).
        """
        if value:
            self.value = value
        return self.value


class LinkAggregator(ABC):
    """
    A LinkAggregator is an object that aggregates the usage and costs across all associated objects
    (e.g. across the links of a Chain object).
    """

    @property
    @abstractmethod
    def history(self) -> list[Record]:
        """A list of Record objects tracking important records/events."""

    @property
    def usage_history(self) -> list[UsageRecord]:
        """Returns all records of type UsageRecord."""
        return [x for x in self.history if isinstance(x, UsageRecord)]

    @property
    def message_history(self) -> list[MessageRecord]:
        """Returns all records of type MessageRecord."""
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
    """
    A Chain object is a collection of `links`. Each link in the chain is a callable, which can be
    either a function or an object that implements the `__call__` method.

    The output of one link serves as the input to the next link in the chain.

    Additionally, each link can track its own history, including messages sent/received and token
    usage/costs, through a `history` property that returns a list of `Record` objects. A Chain
    aggregates and propagates the history of any link that has a `history` property, making it
    convenient to analyze costs or explore intermediate steps in the chain.
    """

    def __init__(self, links: list[Callable[[Any], Any]]):
        self._links = links

    def __getitem__(self, index: int) -> Callable:
        return self._links[index]

    def __len__(self) -> int:
        return len(self._links)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Executes the chain by passing the provided value to the first link. The output of each link
        is passed as the input to the next link, creating a sequential execution of all links in
        the chain.

        The execution continues until all links in the chain have been executed. The final output
        from the last link is then returned.
        """
        if not self._links:
            return None
        result = self._links[0](*args, **kwargs)
        if len(self._links) > 1:
            for link in self._links[1:]:
                result = link(result)
        return result

    @property
    def history(self) -> list[Record]:
        """
        Aggregates the `history` across all links in the Chain. This method ensures that if a link
        is added multiple times to the Chain (e.g. a chat model with multiple steps), the
        underlying Record objects associated with that link's `history` are not duplicated.
        """
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
    A Session is used to aggregate multiple Chain objects. It provides a way to track and manage
    multiple Chains within the same session. When calling a Session, it will execute the last chain
    that was added to the session.
    """

    def __init__(self, chains: list[Chain] | None = None):
        self._chains = chains or []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """
        Calls/starts the chain that was the last added to the session, passing in the corresponding
        arguments.
        """
        if self._chains:
            return self._chains[-1](*args, **kwargs)
        raise ValueError()

    def append(self, chain: Chain) -> None:
        """
        Add or append a new Chain object to the list of chains in the session. If the session
        object is called (i.e. __call__), the session will forward the call to the new chain object
        (i.e. the last chain added in the list).
        """
        self._chains.append(chain)

    def __len__(self) -> int:
        return len(self._chains)

    @property
    def history(self) -> list[Record]:
        """
        Aggregates the `history` across all Chain objects in the Session. This method ensures that
        if a link is added multiple times to the Session, the underlying Record objects associated
        with that link's `history` are not duplicated.
        """
        """
        Aggregates the `history` across all Chains in the session. It ensures that if the same
        object (e.g. chat model) is added multiple times to the Session, that the underlying Record
        objects associated with that object's `history` are not duplicated.
        """
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
    """
    For a given object `obj`, return True if that object has a `history` property and if the
    history property has any Record objects.
    """
    return has_property(obj, property_name='history') and \
        isinstance(obj.history, list) and \
        len(obj.history) > 0 and \
        isinstance(obj.history[0], Record)


class RequestError(Exception):
    """Class that wraps an error when using requests.get()."""

    def __init__(self, status_code: int, reason: str) -> None:
        self.status_code = status_code
        self.reason = reason
