"""TBD."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any
import random
from faker import Faker

from pydantic import BaseModel




class MessageMetaData(BaseModel):
    """
    A MessageMetaData is a single interaction with an LLM (i.e. a prompt and a response. It's used
    to capture additional information about that interaction such as the number of tokens used and
    the corresponding costs.
    """

    prompt: str
    response: str
    # e.g. model-name; not sure how to populate this from sub-classes dynamatically
    # metadata: dict | None
    prompt_tokens: int | None = None
    response_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None

    def __init__(
            self,
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None,
            **data: Any):  # noqa: ANN401
        """
        Init method.

        Args:
            token_counter:
                if provided, the `prompt_tokens`, `response_tokens`, and `total_tokens` fields will
                be calculated from the prompt/response
            cost_per_token: cost_per_token
                if provided, the `cost` field will be calculated from the `total_tokens` field.
            data: misc
        """
        super().__init__(**data)
        if token_counter:
            self.prompt_tokens = token_counter(self.prompt)
            self.response_tokens = token_counter(self.response)
            self.total_tokens = self.prompt_tokens + self.response_tokens
            if cost_per_token:
                self.cost = cost_per_token * self.total_tokens


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
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None) -> None:
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
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token
        self.history: list[MessageMetaData] = []

    @abstractmethod
    def _run(self, prompt: str) -> str:
        """Subclasses should override this function and generate responses from the LLM."""


    def __call__(self, prompt: str) -> str:
        """
        When the object is called it takes a prompt (string) and returns a response (string).

        Args:
            prompt: the string input/question to the model.
        """
        response = self._run(prompt)
        self.history.append(
            MessageMetaData(
                prompt=prompt,
                response=response,
                token_counter=self.token_counter,
                cost_per_token=self.cost_per_token,
            ),
        )
        return response

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
        if not self.token_counter:
            return None
        return sum(x.total_tokens for x in self.history)


    @property
    def total_prompt_tokens(self) -> str:
        """
        Returns the total number of prompt_tokens used by the model during this object's lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        # if there is no token_counter then there won't be a way to calculate the number of tokens
        if not self.token_counter:
            return None
        return sum(x.prompt_tokens for x in self.history)

    @property
    def total_response_tokens(self) -> str:
        """
        Returns the total number of response_tokens used by the model during this object's
        lifetime.

        Returns `None` if the model does not know how to count tokens.
        """
        # if there is no token_counter then there won't be a way to calculate the number of tokens
        if not self.token_counter:
            return None
        return sum(x.response_tokens for x in self.history)

    @property
    def total_cost(self) -> str:
        """
        Returns the total cost associated with usage of the model during this object's lifetime.

        Returns `None` if the model does not know how to count costs.
        """
        # if there is no cost_per_token then there won't be a way to calculate the the cost
        if not self.cost_per_token:
            return None
        return sum(x.cost for x in self.history)


class MockLLM(LLM):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            model_type: str,
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None) -> None:
        super().__init__(
            model_type=model_type,
            token_counter=token_counter,
            cost_per_token=cost_per_token,
        )

    def _run(self, prompt: str) -> str:
        fake = Faker()
        return ' '.join([fake.word() for _ in range(random.randint(10, 100))])


def test_MessageMetaData():  # noqa
    prompt = "This is a prompt."
    response = "This is a response."
    # metadata = {'model_name': 'test-model'}
    message = MessageMetaData(
        prompt=prompt,
        response=response,
        # metadata=metadata,
    )
    assert message.prompt == prompt
    assert message.response == response
    # assert message.metadata == metadata
    assert message.prompt_tokens is None
    assert message.response_tokens is None
    assert message.total_tokens is None
    assert message.cost is None

    message = MessageMetaData(
        prompt=prompt,
        response=response,
        # metadata=metadata,
        token_counter=len,
        cost_per_token=2,
    )
    assert message.prompt == prompt
    assert message.response == response
    # assert message.metadata == metadata
    assert message.prompt_tokens == len(prompt)
    assert message.response_tokens == len(response)
    assert message.total_tokens == len(prompt) + len(response)
    assert message.cost == (len(prompt) + len(response)) * 2


def test_model__no_token_counter_or_costs():  # noqa
    model = MockLLM(model_type='chat', token_counter=None, cost_per_token=None)
    assert model.previous_message is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.total_cost is None
    assert model.total_tokens is None
    assert model.total_prompt_tokens is None
    assert model.total_response_tokens is None

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(model.history) == 1
    message = model.previous_message
    assert isinstance(message, MessageMetaData)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost is None
    assert message.total_tokens is None
    assert message.prompt_tokens is None
    assert message.response_tokens is None

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token is None
    assert model.token_counter is None
    assert model.total_cost is None
    assert model.total_tokens is None
    assert model.total_prompt_tokens is None
    assert model.total_response_tokens is None

    ####
    # second interaction
    ####
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(model.history) == 2
    message = model.previous_message
    assert isinstance(message, MessageMetaData)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost is None
    assert message.total_tokens is None
    assert message.prompt_tokens is None
    assert message.response_tokens is None

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.cost_per_token is None
    assert model.token_counter is None
    assert model.total_cost is None
    assert model.total_tokens is None
    assert model.total_prompt_tokens is None
    assert model.total_response_tokens is None


def test_model__has_token_counter_and_costs():  # noqa
    token_counter = len
    cost_per_token = 3
    model = MockLLM(model_type='chat', token_counter=token_counter, cost_per_token=cost_per_token)
    assert model.previous_message is None
    assert model.previous_prompt is None
    assert model.previous_response is None
    assert model.total_cost == 0
    assert model.total_tokens == 0
    assert model.total_prompt_tokens == 0
    assert model.total_response_tokens == 0

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_prompt_tokens = token_counter(prompt)
    expected_response_tokens = token_counter(response)
    expected_tokens = expected_prompt_tokens + expected_response_tokens
    expected_costs = expected_tokens * cost_per_token

    assert len(model.history) == 1
    message = model.previous_message
    assert isinstance(message, MessageMetaData)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.prompt_tokens == expected_prompt_tokens
    assert message.response_tokens == expected_response_tokens

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.token_counter is token_counter
    assert model.cost_per_token == cost_per_token
    assert model.total_cost == expected_costs
    assert model.total_tokens == expected_tokens
    assert model.total_prompt_tokens == expected_prompt_tokens
    assert model.total_response_tokens == expected_response_tokens

    previous_tokens = expected_tokens
    previous_prompt_tokens = expected_prompt_tokens
    previous_response_tokens = expected_response_tokens
    previous_costs = expected_costs

    ####
    # second interaction
    ####
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    expected_prompt_tokens = token_counter(prompt)
    expected_response_tokens = token_counter(response)
    expected_tokens = expected_prompt_tokens + expected_response_tokens
    expected_costs = expected_tokens * cost_per_token

    assert len(model.history) == 2
    message = model.previous_message
    assert isinstance(message, MessageMetaData)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.prompt_tokens == expected_prompt_tokens
    assert message.response_tokens == expected_response_tokens

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.token_counter is token_counter
    assert model.cost_per_token == cost_per_token
    assert model.total_cost == expected_costs + previous_costs
    assert model.total_tokens == expected_tokens + previous_tokens
    assert model.total_prompt_tokens == expected_prompt_tokens + previous_prompt_tokens
    assert model.total_response_tokens == expected_response_tokens + previous_response_tokens











