"""Contains models."""
from collections.abc import Callable
from llm_chain.base import ChatModel, Document, EmbeddingsRecord, EmbeddingsModel, MemoryBuffer, \
    MessageRecord
from llm_chain.resources import MODEL_COST_PER_1K_TOKENS


class OpenAIEmbeddings(EmbeddingsModel):
    """
    Input: list of documents
    Output: list of documents with embeddings.
    """

    def __init__(
            self,
            model_name: str,
            doc_prep: Callable[[str], str] = lambda x: x.strip().replace('\n', ' '),
            ) -> None:
        """
        TODO.

        NOTE: TODO: cleanup explaination; running this on docs creates side effect and populates
        .embedding property and returns the same list of objects

        Args:
            model_name: e.g. 'text-embedding-ada-002'
            doc_prep:
                function that cleans the text of each doc before creating embeddings.
        """
        super().__init__()
        model_cost_per_1k_tokens = {
            'text-embedding-ada-002': 0.0004,
        }
        self.cost_per_token = model_cost_per_1k_tokens[model_name] / 1000
        self.model_name = model_name
        self.doc_prep = doc_prep

    def _run(self, docs: list[Document]) -> tuple[list[list[float]], EmbeddingsRecord]:
        """TODO."""
        import openai
        texts = [self.doc_prep(x.content) for x in docs]
        response = openai.Embedding.create(input = texts, model=self.model_name)
        total_tokens = response['usage']['total_tokens']
        embeddings = [x['embedding'] for x in response['data']]
        metadata = EmbeddingsRecord(
            metadata={'model_name': self.model_name},
            total_tokens=total_tokens,
            cost=self.cost_per_token * total_tokens,
        )
        return embeddings, metadata


class OpenAIChat(ChatModel):
    """
    Input: prompt/query (string).
    Output: the response (string).
    """

    def __init__(
            self,
            model_name: str,
            temperature: float = 0,
            max_tokens: int = 2000,
            system_message: str = 'You are a helpful assistant.',
            memory_strategy: MemoryBuffer | None = None,  # noqa
            ) -> None:
        """TODO."""
        # TODO: doc string model_name e.g. 'gpt-3.5-turbo'
        # copied from https://github.com/hwchase17/langchain/blob/master/langchain/callbacks/openai_info.py
        super().__init__()
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_token = MODEL_COST_PER_1K_TOKENS[model_name] / 1000
        self.memory_strategy = memory_strategy
        self.system_message = {'role': 'system', 'content': system_message}
        self._previous_memory = None

    def _run(self, prompt: str) -> MessageRecord:
        """
        `openai.ChatCompletion.create` expects a list of messages with various roles (i.e. system,
        user, assistant). This function builds the list of messages based on the history of
        messages and based on an optional 'memory_strategy' that filters the history based on
        it's own logic. The `system_message` is always the first message regardless if a
        `memory_strategy` is passed in.
        """
        import openai
        # initial message
        messages = [self.system_message]
        # build up messages from history
        memory = self._history.copy()
        if self.memory_strategy:
            memory = self.memory_strategy(history=memory)
        for message in memory:
            messages += [
                {'role': 'user', 'content': message.prompt},
                {'role': 'assistant', 'content': message.response},
            ]
        # add latest prompt to messages
        messages += [{'role': 'user', 'content': prompt}]
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        self._previous_memory = messages
        response_message = response['choices'][0]['message'].content
        return MessageRecord(
            prompt=prompt,
            response=response_message,
            metadata={'model_name': self.model_name},
            prompt_tokens=response['usage'].prompt_tokens,
            response_tokens=response['usage'].completion_tokens,
            total_tokens=response['usage'].total_tokens,
            cost=response['usage'].total_tokens * self.cost_per_token,
        )
