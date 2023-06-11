"""Configures the pytests."""
from collections.abc import Callable
import random
from faker import Faker
import numpy as np
import pytest
from dotenv import load_dotenv

from llm_chain.base import ChatModel, Document, EmbeddingsMetaData, EmbeddingsModel, \
    MessageMetaData


class MockChat(ChatModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None,
            return_prompt: str | None = None ) -> None:
        """
        Used to test base classes.

        Args:
            token_counter:
                custom token counter to check total_tokens
            cost_per_token:
                custom costs to check total_costs
            return_prompt:
                if not None, prepends the string to the prompt and returns it as a response
        """
        super().__init__()
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token
        self.return_prompt = return_prompt

    def _run(self, prompt: str) -> MessageMetaData:
        if self.return_prompt:
            response = self.return_prompt + prompt
        else:
            fake = Faker()
            response = ' '.join([fake.word() for _ in range(random.randint(10, 100))])
        prompt_tokens = self.token_counter(prompt) if self.token_counter else None
        response_tokens = self.token_counter(response) if self.token_counter else None
        total_tokens = prompt_tokens + response_tokens if self.token_counter else None
        cost = total_tokens * self.cost_per_token if self.cost_per_token else None
        return MessageMetaData(
            prompt=prompt,
            response=response,
            total_tokens=total_tokens,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            cost=cost,
            metadata={'model_name': 'mock'},
        )


class MockEmbeddings(EmbeddingsModel):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None) -> None:
        super().__init__()
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token

    def _run(self, docs: list[Document]) -> tuple[list[Document], EmbeddingsMetaData]:
        response = [np.random.rand(5) for _ in docs]
        for doc, embedding in zip(docs, response):
            doc.embedding = embedding
        total_tokens = sum(self.token_counter(x.content) for x in docs) \
            if self.token_counter else None
        cost = total_tokens * self.cost_per_token if self.cost_per_token else None
        return docs, EmbeddingsMetaData(
            total_tokens=total_tokens,
            cost=cost,
        )

@pytest.fixture(scope="session", autouse=True)
def load_env_vars():  # noqa
    load_dotenv()
