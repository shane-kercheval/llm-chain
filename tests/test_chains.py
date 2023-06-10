"""TBD."""
# from llm_chain.base import Document, EmbeddingsMetaData, EmbeddingsModel
import numpy as np
from llm_chain.base import Document, EmbeddingsMetaData, MessageMetaData
from llm_chain.memory import MemoryBufferWindow
from llm_chain.models import OpenAIChat, OpenAIEmbeddings
from tests.conftest import MockChat, MockEmbeddings


def test_ChatModel__no_token_counter_or_costs():  # noqa
    model = MockChat(token_counter=None, cost_per_token=None)
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

def test_ChatModel__has_token_counter_and_costs():  # noqa
    token_counter = len
    cost_per_token = 3
    model = MockChat(token_counter=token_counter, cost_per_token=cost_per_token)
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

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
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
    ####
    prompt = "This is another question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
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

def test_OpenAIChat__MemoryBufferWindow_0():  # noqa
    openai_llm = OpenAIChat(
        model_name='gpt-3.5-turbo',
        memory_strategy=MemoryBufferWindow(last_n_messages=0),
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
        memory_strategy=MemoryBufferWindow(last_n_messages=1),
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

def test_EmbeddingsModel__no_costs():  # noqa
    model = MockEmbeddings(token_counter=len, cost_per_token=None)
    assert model.total_cost is None
    assert model.total_tokens is None

    ####
    # first interaction
    ####
    doc_content_0 = "This is a doc."
    doc_content_1 = "This is a another doc."
    docs = [
        Document(content=doc_content_0),
        Document(content=doc_content_1),
    ]
    new_docs = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], Document)
    assert len(new_docs) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1
    assert new_docs[0].content == doc_content_0
    assert new_docs[1].content == doc_content_1
    assert all(isinstance(x.embedding, np.ndarray) for x in new_docs)

    assert len(model.history) == 1
    last_metadata = model.history[0]
    assert isinstance(last_metadata, EmbeddingsMetaData)
    assert last_metadata.total_tokens == expected_tokens
    assert last_metadata.cost is None

    assert model.total_tokens == expected_tokens
    assert model.total_cost is None

    previous_tokens = model.total_tokens

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    new_docs = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], Document)
    assert len(new_docs) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3
    assert new_docs[0].content == doc_content_2
    assert new_docs[1].content == doc_content_3
    assert all(isinstance(x.embedding, np.ndarray) for x in new_docs)

    assert len(model.history) == 2
    first_meata = model.history[0]
    assert isinstance(first_meata, EmbeddingsMetaData)
    assert first_meata.total_tokens == previous_tokens
    assert first_meata.cost is None

    last_metadata = model.history[1]
    assert isinstance(last_metadata, EmbeddingsMetaData)
    assert last_metadata.total_tokens == expected_tokens
    assert last_metadata.cost is None

    assert model.total_tokens == previous_tokens + expected_tokens
    assert model.total_cost is None

def test_EmbeddingsModel__with_costs():  # noqa
    cost_per_token = 3
    model = MockEmbeddings(token_counter=len, cost_per_token=cost_per_token)
    assert model.total_cost is None
    assert model.total_tokens is None

    ####
    # first interaction
    ####
    doc_content_0 = "This is a doc."
    doc_content_1 = "This is a another doc."
    docs = [
        Document(content=doc_content_0),
        Document(content=doc_content_1),
    ]
    new_docs = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    expected_cost = expected_tokens * cost_per_token
    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], Document)
    assert len(new_docs) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1
    assert new_docs[0].content == doc_content_0
    assert new_docs[1].content == doc_content_1
    assert all(isinstance(x.embedding, np.ndarray) for x in new_docs)

    assert len(model.history) == 1
    last_metadata = model.history[0]
    assert isinstance(last_metadata, EmbeddingsMetaData)
    assert last_metadata.total_tokens == expected_tokens
    assert last_metadata.cost == expected_cost

    assert model.total_tokens == expected_tokens
    assert model.total_cost == expected_cost

    previous_tokens = model.total_tokens
    previous_cost = model.total_cost

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    new_docs = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    expected_cost = expected_tokens * cost_per_token
    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], Document)
    assert len(new_docs) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3
    assert new_docs[0].content == doc_content_2
    assert new_docs[1].content == doc_content_3
    assert all(isinstance(x.embedding, np.ndarray) for x in new_docs)

    assert len(model.history) == 2
    first_meata = model.history[0]
    assert isinstance(first_meata, EmbeddingsMetaData)
    assert first_meata.total_tokens == previous_tokens
    assert first_meata.cost == previous_cost

    last_metadata = model.history[1]
    assert isinstance(last_metadata, EmbeddingsMetaData)
    assert last_metadata.total_tokens == expected_tokens
    assert last_metadata.cost == expected_cost

    assert model.total_tokens == previous_tokens + expected_tokens
    assert model.total_cost == previous_cost + expected_cost


def test_OpenAIEmbeddings():  # noqa
    model = OpenAIEmbeddings(model_name='text-embedding-ada-002')
    assert model.total_cost is None
    assert model.total_tokens is None

    ####
    # first interaction
    ####
    doc_content_0 = "This is a doc."
    doc_content_1 = "This is a another doc."
    docs = [
        Document(content=doc_content_0),
        Document(content=doc_content_1),
    ]
    new_docs = model(docs)
    expected_cost = model.history[0].total_tokens * model.cost_per_token
    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], Document)
    assert len(new_docs) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1
    assert new_docs[0].content == doc_content_0
    assert new_docs[1].content == doc_content_1
    assert all(isinstance(x.embedding, list) for x in new_docs)
    assert all(len(x.embedding) > 100 for x in new_docs)

    assert len(model.history) == 1
    last_metadata = model.history[0]
    assert isinstance(last_metadata, EmbeddingsMetaData)
    assert last_metadata.total_tokens > 0
    assert last_metadata.cost == expected_cost

    assert model.total_tokens == last_metadata.total_tokens
    assert model.total_cost == expected_cost

    previous_tokens = model.total_tokens
    previous_cost = model.total_cost

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    new_docs = model(docs)
    expected_cost = model.history[1].total_tokens * model.cost_per_token
    assert isinstance(new_docs, list)
    assert isinstance(new_docs[0], Document)
    assert len(new_docs) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3
    assert new_docs[0].content == doc_content_2
    assert new_docs[1].content == doc_content_3
    assert all(isinstance(x.embedding, list) for x in new_docs)

    assert len(model.history) == 2
    first_meata = model.history[0]
    assert isinstance(first_meata, EmbeddingsMetaData)
    assert first_meata.total_tokens == previous_tokens
    assert first_meata.cost == previous_cost

    last_metadata = model.history[1]
    assert isinstance(last_metadata, EmbeddingsMetaData)
    assert last_metadata.total_tokens > 0
    assert last_metadata.cost == expected_cost

    assert model.total_tokens == previous_tokens + last_metadata.total_tokens
    assert model.total_cost == previous_cost + expected_cost

# TODO: implement rety (refactor call to openai.ChatCompletion.create)
# TODO: test embeddings when text size is larger than context
# TODO: test MemoryBufferTokenWindow
