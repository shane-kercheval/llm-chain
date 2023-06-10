"""tests llm_chain/memory.py."""

from llm_chain.base import MessageMetaData
from llm_chain.memory import MemoryBufferMessageWindow
from llm_chain.models import OpenAIChat


def test_OpenAIChat__MemoryBufferWindow_0():  # noqa
    openai_llm = OpenAIChat(
        model_name='gpt-3.5-turbo',
        memory_strategy=MemoryBufferMessageWindow(last_n_messages=0),
    )
    assert openai_llm.previous_message is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.total_cost is None
    assert openai_llm.total_tokens is None
    assert openai_llm.total_prompt_tokens is None
    assert openai_llm.total_response_tokens is None

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is shane. What is my name?"
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt

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

    # previous_prompt = prompt
    # previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_prompt_tokens = message.prompt_tokens
    previous_response_tokens = message.response_tokens

    ####
    # second interaction
    # this shouldn't be any different
    ####
    prompt = "What is my name?"
    response = openai_llm(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert len(openai_llm._previous_memory) == 2  # system/user
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == prompt

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

def test_OpenAIChat__MemoryBufferWindow_1():  # noqa
    openai_llm = OpenAIChat(
        model_name='gpt-3.5-turbo',
        memory_strategy=MemoryBufferMessageWindow(last_n_messages=1),
    )
    assert openai_llm.previous_message is None
    assert openai_llm.previous_prompt is None
    assert openai_llm.previous_response is None
    assert openai_llm.total_cost is None
    assert openai_llm.total_tokens is None
    assert openai_llm.total_prompt_tokens is None
    assert openai_llm.total_response_tokens is None

    ####
    # first interaction
    # this shouldn't be any different
    ####
    prompt = "Hi my name is shane. What is my name?"
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt

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

    previous_prompt = prompt
    previous_response = response
    previous_cost = message.cost
    previous_total_tokens = message.total_tokens
    previous_prompt_tokens = message.prompt_tokens
    previous_response_tokens = message.response_tokens

    ####
    # second interaction
    # this shouldn't be any different
    ####
    prompt = "What is my name?"
    response = openai_llm(prompt)
    assert 'shane' in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt

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

    previous_prompt = prompt
    previous_response = response

    ####
    # third interaction
    # this shouldn't be any different
    ####
    prompt = "What is today's date?"
    response = openai_llm(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    # The last message should contain shane, but not this one
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert 'shane' in openai_llm._previous_memory[2]['content'].lower()
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt
    assert len(openai_llm._previous_memory) == 4

    assert len(openai_llm.history) == 3
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

    previous_prompt = prompt
    previous_response = response

    ####
    # 4th interaction
    # this shouldn't contain the name shane because the last interaction was the first that didn't
    ####
    prompt = "What is today's date?"
    response = openai_llm(prompt)
    assert 'shane' not in response.lower()
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt
    # still 4 because we are only keeping 1 message
    # (1)system + (2)previous question + (3)previous answer + (4)new question
    assert len(openai_llm._previous_memory) == 4

    assert len(openai_llm.history) == 4
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

# TODO: test MemoryBufferTokenWindow
