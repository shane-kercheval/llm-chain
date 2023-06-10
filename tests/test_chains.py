"""TBD."""
from collections.abc import Callable
import random
from faker import Faker
from llm_chain.base import LLM, MessageMetaData
from llm_chain.models import OpenAIChat


class MockLLM(LLM):
    """Used for unit tests to mock the behavior of an LLM."""

    def __init__(
            self,
            model_type: str,
            token_counter: Callable[[str], int] | None = None,
            cost_per_token: float | None = None) -> None:
        super().__init__(model_type=model_type)
        self.token_counter = token_counter
        self.cost_per_token = cost_per_token

    def _run(self, prompt: str) -> MessageMetaData:
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
    assert message.metadata == {'model_name': 'mock'}
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
    assert message.metadata == {'model_name': 'mock'}
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

def test_OpenAIChat():  # noqa
    from dotenv import load_dotenv
    load_dotenv()

    openai_llm = OpenAIChat(model_name='gpt-3.5-turbo')
    assert openai_llm.previous_message is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.total_cost is None
    assert openai_llm.total_tokens is None
    assert openai_llm.total_prompt_tokens is None
    assert openai_llm.total_response_tokens is None

    ####
    # first interaction
    ####
    prompt = "This is a question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(openai_llm.history) == 1
    message = openai_llm.previous_message
    assert isinstance(message, MessageMetaData)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'gpt-3.5-turbo'}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == 0.002 / 1000
    assert openai_llm.total_cost == message.cost
    assert openai_llm.total_tokens == message.total_tokens
    assert openai_llm.total_prompt_tokens == message.prompt_tokens
    assert openai_llm.total_response_tokens == message.response_tokens

    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_prompt_tokens = message.prompt_tokens
    previous_response_tokens = message.response_tokens

    ####
    # second interaction
    ####
    prompt = "This is another question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(openai_llm.history) == 2
    message = openai_llm.previous_message
    assert isinstance(message, MessageMetaData)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'gpt-3.5-turbo'}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == 0.002 / 1000
    assert openai_llm.total_cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.total_prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.total_response_tokens == previous_response_tokens + message.response_tokens
