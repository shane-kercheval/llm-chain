"""Tests the utilities.py file."""
from time import sleep
import re
import pytest
import openai
from llm_chain.utilities import Timer, create_hash, num_tokens, num_tokens_from_messages, \
    retry_handler

def test_timer_seconds():  # noqa
    with Timer() as timer:
        sleep(1.1)

    assert timer.interval
    assert re.match(pattern=r'1\.\d+ seconds', string=timer.formatted())
    assert str(timer) == timer.formatted()

    with pytest.raises(ValueError):  # noqa
        timer.formatted(units='days')


def test_create_hash():  # noqa
    value_a = create_hash('Test value 1')
    assert value_a
    value_b = create_hash('Test value 2')
    assert value_b
    assert value_a != value_b
    value_c = create_hash('Test value 1')
    assert value_c == value_a


def test_num_tokens():  # noqa
    assert num_tokens(model_name='gpt-3.5-turbo', value="This should be six tokens.") == 6


def test_num_tokens_from_messages():  # noqa
    # copied from https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    example_messages = [
        {
            "role": "system",
            "content": "You are a helpful, pattern-following assistant that translates corporate jargon into plain English.",  # noqa
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "New synergies will help drive top-line growth.",
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Things working well together will increase revenue.",
        },
        {
            "role": "system",
            "name": "example_user",
            "content": "Let's circle back when we have more bandwidth to touch base on opportunities for increased leverage.",  # noqa
        },
        {
            "role": "system",
            "name": "example_assistant",
            "content": "Let's talk later when we're less busy about how to do better.",
        },
        {
            "role": "user",
            "content": "This late pivot means we don't have time to boil the ocean for the client deliverable.",  # noqa
        },
    ]
    model_name = 'gpt-3.5-turbo'
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=example_messages,
        temperature=0,
        max_tokens=1,  # we're only counting input tokens here, so let's not waste tokens on output
    )
    expected_value = response["usage"]["prompt_tokens"]
    actual_value = num_tokens_from_messages(model_name=model_name, messages=example_messages)
    assert expected_value == actual_value

    # above we checked that the numbers match exactly from what OpenAI returns;
    # here, let's just check that the other models run and return >0 to avoid API calls
    assert num_tokens_from_messages(model_name='gpt-3.5-turbo-0301', messages=example_messages) > 0
    assert num_tokens_from_messages(model_name='gpt-4', messages=example_messages) > 0
    assert num_tokens_from_messages(model_name='gpt-4-0314', messages=example_messages) > 0
    with pytest.raises(NotImplementedError):
        num_tokens_from_messages(model_name='<not implemented>', messages=example_messages)

def test_retry_handler():  # noqa
    r = retry_handler()
    actual_value = r(
        lambda x, y: (x, y),
        x='A',
        y='B',
    )
    assert actual_value == ('A', 'B')
