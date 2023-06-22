"""tests llm_chain/models.py."""
from llm_chain.base import Document, EmbeddingsRecord, MessageRecord, StreamingRecord
from llm_chain.models import OpenAIChat, OpenAIEmbeddings
from llm_chain.resources import MODEL_COST_PER_TOKEN
from tests.conftest import MockChat, MockRandomEmbeddings


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

    assert len(model._history) == 1
    message = model.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': 'mock'}
    assert message.cost is None
    assert message.total_tokens is None
    assert message.prompt_tokens is None
    assert message.response_tokens is None
    assert message.uuid
    assert message.timestamp

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
    previous_message = message
    prompt = "This is another question."
    response = model(prompt)
    assert isinstance(response, str)
    assert len(response) > 1

    assert len(model._history) == 2
    message = model.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.metadata == {'model_name': 'mock'}
    assert message.response == response
    assert message.cost is None
    assert message.total_tokens is None
    assert message.prompt_tokens is None
    assert message.response_tokens is None
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

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

    assert len(model._history) == 1
    message = model.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.prompt_tokens == expected_prompt_tokens
    assert message.response_tokens == expected_response_tokens
    assert message.uuid
    assert message.timestamp

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
    previous_message = message

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

    assert len(model._history) == 2
    message = model.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.cost == expected_costs
    assert message.total_tokens == expected_tokens
    assert message.prompt_tokens == expected_prompt_tokens
    assert message.response_tokens == expected_response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert model.previous_prompt == prompt
    assert model.previous_response == response
    assert model.token_counter is token_counter
    assert model.cost_per_token == cost_per_token
    assert model.total_cost == expected_costs + previous_costs
    assert model.total_tokens == expected_tokens + previous_tokens
    assert model.total_prompt_tokens == expected_prompt_tokens + previous_prompt_tokens
    assert model.total_response_tokens == expected_response_tokens + previous_response_tokens

def test_OpenAIChat():  # noqa
    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(model_name=model_name)
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

    assert len(openai_llm._history) == 1
    message = openai_llm.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
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
    previous_message = message

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

    assert len(openai_llm._history) == 2
    message = openai_llm.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.total_cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.total_prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.total_response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming():  # noqa
    """Test the same thing as above but for streaming. All usage and history should be the same."""
    callback_response = ''
    def streaming_callback(record: StreamingRecord) -> None:
        nonlocal callback_response
        callback_response += record.response

    model_name = 'gpt-3.5-turbo'
    openai_llm = OpenAIChat(model_name=model_name, streaming_callback=streaming_callback)
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
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[0]['content'] == 'You are a helpful assistant.'
    assert openai_llm._previous_memory[-1]['role'] == 'user'
    assert openai_llm._previous_memory[-1]['content'] == prompt

    assert len(openai_llm._history) == 1
    message = openai_llm.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
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
    previous_message = message

    ####
    # second interaction
    ####
    callback_response = ''
    prompt = "This is another question."
    response = openai_llm(prompt)
    assert isinstance(response, str)
    assert len(response) > 1
    assert response == callback_response

    # previous memory is the input to ChatGPT
    assert openai_llm._previous_memory[0]['role'] == 'system'
    assert openai_llm._previous_memory[1]['role'] == 'user'
    assert openai_llm._previous_memory[1]['content'] == previous_prompt
    assert openai_llm._previous_memory[2]['role'] == 'assistant'
    assert openai_llm._previous_memory[2]['content'] == previous_response
    assert openai_llm._previous_memory[3]['role'] == 'user'
    assert openai_llm._previous_memory[3]['content'] == prompt

    assert len(openai_llm._history) == 2
    message = openai_llm.previous_message
    assert isinstance(message, MessageRecord)
    assert message.prompt == prompt
    assert message.response == response
    assert message.metadata == {'model_name': model_name}
    assert message.cost > 0
    assert message.prompt_tokens > 0
    assert message.response_tokens > 0
    assert message.total_tokens == message.prompt_tokens + message.response_tokens
    assert message.uuid
    assert message.uuid != previous_message.uuid
    assert message.timestamp

    assert openai_llm.previous_prompt == prompt
    assert openai_llm.previous_response == response
    assert openai_llm.cost_per_token == MODEL_COST_PER_TOKEN[model_name]
    assert openai_llm.total_cost == previous_cost + message.cost
    assert openai_llm.total_tokens == previous_total_tokens + message.total_tokens
    assert openai_llm.total_prompt_tokens == previous_prompt_tokens + message.prompt_tokens
    assert openai_llm.total_response_tokens == previous_response_tokens + message.response_tokens

def test_OpenAIChat_streaming_response_matches_non_streaming():  # noqa
    """
    Test that we get the same final response and usage data when streaming vs not streaming.
    Additionally test that the response we get in the streaming callback matches the overall
    response.
    """
    question = "Explain what a large language model is in a single sentence."
    model_name = 'gpt-3.5-turbo'
    non_streaming_chat = OpenAIChat(
        model_name=model_name,
        temperature=0,
        )
    non_streaming_response = non_streaming_chat(question)

    callback_response = ''
    def streaming_callback(record: StreamingRecord) -> None:
        nonlocal callback_response
        callback_response += record.response

    streaming_chat = OpenAIChat(
        model_name=model_name,
        temperature=0,
        streaming_callback=streaming_callback,
    )
    streaming_response  = streaming_chat(question)
    assert non_streaming_response == streaming_response
    assert non_streaming_response == callback_response
    assert non_streaming_chat.total_prompt_tokens == streaming_chat.total_prompt_tokens
    assert non_streaming_chat.total_response_tokens == streaming_chat.total_response_tokens
    assert non_streaming_chat.total_tokens == streaming_chat.total_tokens

def test_EmbeddingsModel__no_costs():  # noqa
    model = MockRandomEmbeddings(token_counter=len, cost_per_token=None)
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
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1

    assert len(model._history) == 1
    first_record = model._history[0]
    assert isinstance(first_record, EmbeddingsRecord)
    assert first_record.total_tokens == expected_tokens
    assert first_record.cost is None
    assert first_record.uuid
    assert first_record.timestamp

    assert model.total_tokens == expected_tokens
    assert model.total_cost is None

    previous_tokens = model.total_tokens
    previous_record = first_record

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3

    assert len(model._history) == 2
    first_record = model._history[0]
    assert isinstance(first_record, EmbeddingsRecord)
    assert first_record.total_tokens == previous_tokens
    assert first_record.cost is None
    assert first_record.uuid
    assert first_record.uuid == previous_record.uuid
    assert first_record.timestamp

    second_record = model._history[1]
    assert isinstance(second_record, EmbeddingsRecord)
    assert second_record.total_tokens == expected_tokens
    assert second_record.cost is None
    assert second_record.uuid
    assert second_record.uuid != previous_record.uuid
    assert second_record.timestamp

    assert model.total_tokens == previous_tokens + expected_tokens
    assert model.total_cost is None

def test_EmbeddingsModel__with_costs():  # noqa
    cost_per_token = 3
    model = MockRandomEmbeddings(token_counter=len, cost_per_token=cost_per_token)
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
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    expected_cost = expected_tokens * cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)

    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1

    assert len(model._history) == 1
    first_record = model._history[0]
    assert isinstance(first_record, EmbeddingsRecord)
    assert first_record.total_tokens == expected_tokens
    assert first_record.cost == expected_cost
    assert first_record.uuid
    assert first_record.timestamp

    assert model.total_tokens == expected_tokens
    assert model.total_cost == expected_cost

    previous_tokens = model.total_tokens
    previous_cost = model.total_cost
    previous_record = first_record

    ####
    # second interaction
    ####
    doc_content_2 = "This is a doc for a second call."
    doc_content_3 = "This is a another doc for a second call."
    docs = [
        Document(content=doc_content_2),
        Document(content=doc_content_3),
    ]
    embeddings = model(docs)
    expected_tokens = sum(len(x.content) for x in docs)
    expected_cost = expected_tokens * cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert all(isinstance(x, list) for x in embeddings)

    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3

    assert len(model._history) == 2
    first_record = model._history[0]
    assert isinstance(first_record, EmbeddingsRecord)
    assert first_record.total_tokens == previous_tokens
    assert first_record.cost == previous_cost
    assert first_record.uuid
    assert first_record.uuid == previous_record.uuid
    assert first_record.timestamp

    second_record = model._history[1]
    assert isinstance(second_record, EmbeddingsRecord)
    assert second_record.total_tokens == expected_tokens
    assert second_record.cost == expected_cost
    assert second_record.uuid
    assert second_record.uuid != previous_record.uuid
    assert second_record.timestamp

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
    embeddings = model(docs)
    expected_cost = model._history[0].total_tokens * model.cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_0
    assert docs[1].content == doc_content_1
    assert all(isinstance(x, list) for x in embeddings)
    assert all(len(x) > 100 for x in embeddings)

    assert len(model._history) == 1
    previous_record = model._history[0]
    assert isinstance(previous_record, EmbeddingsRecord)
    assert previous_record.total_tokens > 0
    assert previous_record.cost == expected_cost
    assert previous_record.uuid
    assert previous_record.timestamp
    assert previous_record.metadata['model_name'] == 'text-embedding-ada-002'

    assert model.total_tokens == previous_record.total_tokens
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
    embeddings = model(docs)
    expected_cost = model._history[1].total_tokens * model.cost_per_token
    assert isinstance(embeddings, list)
    assert isinstance(embeddings[0][0], float)
    assert len(embeddings) == len(docs)
    assert docs[0].content == doc_content_2
    assert docs[1].content == doc_content_3
    assert all(isinstance(x, list) for x in embeddings)

    assert len(model._history) == 2
    first_record = model._history[0]
    assert isinstance(first_record, EmbeddingsRecord)
    assert first_record.total_tokens == previous_tokens
    assert first_record.cost == previous_cost
    assert first_record.uuid == previous_record.uuid
    assert first_record.timestamp == previous_record.timestamp
    assert first_record.metadata['model_name'] == 'text-embedding-ada-002'

    previous_record = model._history[1]
    assert isinstance(previous_record, EmbeddingsRecord)
    assert previous_record.total_tokens > 0
    assert previous_record.cost == expected_cost
    assert previous_record.uuid
    assert previous_record.uuid != first_record.uuid
    assert previous_record.timestamp
    assert previous_record.metadata['model_name'] == 'text-embedding-ada-002'

    assert model.total_tokens == previous_tokens + previous_record.total_tokens
    assert model.total_cost == previous_cost + expected_cost
