"""Contains models."""
from collections.abc import Callable
from llm_chain.base import ChatModel, Document, EmbeddingsRecord, EmbeddingsModel, MemoryBuffer, \
    MessageRecord, StreamingEvent
from llm_chain.resources import MODEL_COST_PER_TOKEN
from llm_chain.utilities import num_tokens, num_tokens_from_messages, retry_handler


class OpenAIEmbeddings(EmbeddingsModel):
    """
    A convenient wrapper around the OpenAI Embeddings model. When you invoke this object with a
    list of Document objects, it will return a tuple. This tuple consists of two elements:
    1. The embeddings, which are represented as a list where each item corresponds to a Document
    and contains the embedding (a list of floats).
    2. An `EmbeddingsRecord` object, which track of costs and other relevant metadata.
    """

    def __init__(
            self,
            model_name: str,
            doc_prep: Callable[[str], str] = lambda x: x.strip().replace('\n', ' '),
            timeout: int = 10,
            ) -> None:
        """
        Args:
            model_name:
                e.g. 'text-embedding-ada-002'
            doc_prep:
                function that cleans the text of each doc before creating embeddings.
            timeout:
                timeout value passed to OpenAI model.
        """
        super().__init__()
        self.model_name = model_name
        self.doc_prep = doc_prep
        self.timeout = timeout

    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingsRecord]:
        import openai
        texts = [self.doc_prep(x.content) for x in docs]
        response = retry_handler()(
            openai.Embedding.create,
            input = texts,
            model=self.model_name,
            timeout=self.timeout,
        )
        total_tokens = response['usage']['total_tokens']
        embeddings = [x['embedding'] for x in response['data']]
        metadata = EmbeddingsRecord(
            metadata={'model_name': self.model_name},
            total_tokens=total_tokens,
            cost=self.cost_per_token * total_tokens,
        )
        return embeddings, metadata

    @property
    def cost_per_token(self) -> float:
        """
        Returns a float corresponding to the cost-per-token for the corresponding model.
        We need to dynamically look this up since the model_name can change over the course of the
        object's lifetime.
        """
        return MODEL_COST_PER_TOKEN[self.model_name]

class OpenAIChat(ChatModel):
    """
    Wrapper around the OpenAI chat model (i.e. https://api.openai.com/v1/chat/completions
    endpoint). More info here: https://platform.openai.com/docs/api-reference/chat.
    """

    def __init__(
            self,
            model_name: str,
            temperature: float = 0,
            max_tokens: int = 2000,
            system_message: str = 'You are a helpful assistant.',
            streaming_callback: Callable[[StreamingEvent], None] | None = None,
            memory_strategy: MemoryBuffer | Callable[[list[MessageRecord]], list[MessageRecord]] | None = None,  # noqa: E501
            timeout: int = 10,
            ) -> None:
        """
        Args:
            model_name:
                e.g. 'gpt-3.5-turbo'
            temperature:
                "What sampling temperature to use, between 0 and 2. Higher values like 0.8 will
                make the output more random, while lower values like 0.2 will make it more focused
                and deterministic."
            max_tokens:
                The maximum number of tokens to generate in the chat completion.
                The total length of input tokens and generated tokens is limited by the model's
                context length.
            system_message:
                The content of the message associated with the "system" `role`.
            streaming_callback:
                Callable that takes a StreamingEvent object, which contains the streamed token (in
                the `response` property and perhaps other metadata.
            memory_strategy:
                MemoryBuffer object (or callable that takes a list of MessageRecord objects and
                returns a list of MessageRecord objects. The underlying logic should return the
                messages sent to the OpenAI model.
            timeout:
                timeout value passed to OpenAI model.
        """
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_strategy = memory_strategy
        self.system_message = {'role': 'system', 'content': system_message}
        self.streaming_callback = streaming_callback
        self.timeout = timeout
        self._previous_memory = None

    def _run(self, prompt: str) -> MessageRecord:
        """
        `openai.ChatCompletion.create` expects a list of messages with various roles (i.e. system,
        user, assistant). This function builds the list of messages based on the history of
        messages and based on an optional 'memory_strategy' that filters the history based on
        it's own logic. The `system_message` is always the first message regardless if a
        `memory_strategy` is passed in.

        The use of a streaming callback does not change the output returned from calling the object
        (i.e. a MessageRecord object).
        """
        import openai
        # build up messages from history
        memory = self.history.copy()
        if self.memory_strategy:
            memory = self.memory_strategy(history=memory)

        # initial message; always keep system message regardless of memory_strategy
        messages = [self.system_message]
        for message in memory:
            messages += [
                {'role': 'user', 'content': message.prompt},
                {'role': 'assistant', 'content': message.response},
            ]
        # add latest prompt to messages
        messages += [{'role': 'user', 'content': prompt}]
        if self.streaming_callback:
            response = retry_handler()(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
                stream=True,
            )
            # extract the content/token from the streaming response and send to the callback
            # build up the message so that we can calculate usage/costs and send back the same
            # MessageRecord response that we would return if we weren't streaming
            def get_delta(chunk):  # noqa
                delta = chunk['choices'][0]['delta']
                if 'content' in delta:
                    return delta['content']
                return None
            response_message = ''
            for chunk in response:
                delta = get_delta(chunk)
                if delta:
                    self.streaming_callback(StreamingEvent(response=delta))
                    response_message += delta

            prompt_tokens = num_tokens_from_messages(model_name=self.model_name, messages=messages)
            completion_tokens = num_tokens(model_name=self.model_name, value=response_message)
            total_tokens = prompt_tokens + completion_tokens
        else:
            response = retry_handler()(
                openai.ChatCompletion.create,
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                timeout=self.timeout,
            )
            response_message = response['choices'][0]['message'].content
            prompt_tokens = response['usage'].prompt_tokens
            completion_tokens = response['usage'].completion_tokens
            total_tokens = response['usage'].total_tokens

        self._previous_memory = messages
        cost = (prompt_tokens * self.cost_per_token['input']) + \
            (completion_tokens * self.cost_per_token['output'])

        return MessageRecord(
            prompt=prompt,
            response=response_message,
            metadata={'model_name': self.model_name},
            prompt_tokens=prompt_tokens,
            response_tokens=completion_tokens,
            total_tokens=total_tokens,
            cost=cost,
        )

    @property
    def cost_per_token(self) -> dict:
        """
        Returns a dictionary containing 'input' and 'output' keys each containing a float
        corresponding to the cost-per-token for the corresponding token type and model.
        We need to dynamically look this up since the model_name can change over the course of the
        object's lifetime.
        """
        return MODEL_COST_PER_TOKEN[self.model_name]
