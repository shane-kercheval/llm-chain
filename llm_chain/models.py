"""Contains models."""
from collections.abc import Callable
from llm_chain.base import ChatModel, Document, EmbeddingsMetaData, EmbeddingsModel, \
    MemoryBuffer, MessageMetaData


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

    def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingsMetaData]:
        """
        TODO.

        Populates the `embedding` property on the Document.
        """
        import openai
        texts = [self.doc_prep(x.content) for x in docs]
        response = openai.Embedding.create(input = texts, model=self.model_name)
        total_tokens = response['usage']['total_tokens']
        embeddings = [x['embedding'] for x in response['data']]
        for doc, embedding in zip(docs, embeddings):
            doc.embedding = embedding
        metadata = EmbeddingsMetaData(
            total_tokens=total_tokens,
            cost=self.cost_per_token * total_tokens,
        )
        return docs, metadata


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
        model_cost_per_1k_tokens = {
            'gpt-4': 0.03,
            'gpt-4-0314': 0.03,
            'gpt-4-completion': 0.06,
            'gpt-4-0314-completion': 0.06,
            'gpt-4-32k': 0.06,
            'gpt-4-32k-0314': 0.06,
            'gpt-4-32k-completion': 0.12,
            'gpt-4-32k-0314-completion': 0.12,
            'gpt-3.5-turbo': 0.002,
            'gpt-3.5-turbo-0301': 0.002,
            'text-ada-001': 0.0004,
            'ada': 0.0004,
            'text-babbage-001': 0.0005,
            'babbage': 0.0005,
            'text-curie-001': 0.002,
            'curie': 0.002,
            'text-davinci-003': 0.02,
            'text-davinci-002': 0.02,
            'code-davinci-002': 0.02,
            'ada-finetuned': 0.0016,
            'babbage-finetuned': 0.0024,
            'curie-finetuned': 0.012,
            'davinci-finetuned': 0.12,
        }
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.cost_per_token = model_cost_per_1k_tokens[model_name] / 1000
        self.memory_strategy = memory_strategy
        self.system_message = {'role': 'system', 'content': system_message}
        self._previous_memory = None

    def _run(self, prompt: str) -> MessageMetaData:
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
        memory = self.history.copy()
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
        return MessageMetaData(
            prompt=prompt,
            response=response_message,
            metadata={'model_name': self.model_name},
            prompt_tokens=response['usage'].prompt_tokens,
            response_tokens=response['usage'].completion_tokens,
            total_tokens=response['usage'].total_tokens,
            cost=response['usage'].total_tokens * self.cost_per_token,
        )
