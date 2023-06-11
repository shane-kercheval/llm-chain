"""TODO."""
from llm_chain.base import Document
from llm_chain.chains import Chain, _has_property
from tests.conftest import MockChat, MockEmbeddings


def test_chain():  # noqa
    # test empty chain
    chain = Chain(chain=[])
    assert chain() is None
    assert chain('param') is None
    assert chain(value=1) is None

    # test chain with one link
    chain = Chain(chain=[lambda x: x * 2])
    assert chain(10) == 20
    assert chain([1]) == [1, 1]

    # test chain with two links and test args/kwargs
    def add_one(value: int) -> int:
        return value + 1

    chain = Chain(chain=[add_one, lambda x: x * 2])
    assert chain(10) == 22
    assert chain(value=10) == 22

    # test with three links
    chain = Chain(chain=[add_one, lambda x: x * 2, add_one])
    assert chain(10) == 23
    assert chain(value=10) == 23

def test_has_property():  # noqa
    chat = MockChat()
    lambda_func = lambda x: x  # noqa
    assert _has_property(obj=chat, property_name='total_tokens')
    assert not _has_property(obj=lambda_func, property_name='total_tokens')
    assert not _has_property(obj=chat, property_name='does_not_have')
    assert not _has_property(obj=lambda_func, property_name='does_not_have')

def test_Chain_with_MockChat():  # noqa
    prompt = "Here's a question."
    first_response = "Response: " + prompt
    second_prompt = "Question: " + first_response
    second_response = "Response: " + second_prompt

    chat = MockChat(return_prompt="Response: ")  # this Chat returns the "Response: " + prompt
    chain = Chain(chain=[chat, lambda x: "Question: " + x, chat])
    result = chain(prompt)
    # the final result should be the response returned by the second invokation of chat()
    assert result == second_response

    # check that the prompts/responses got propegated through the chain
    assert len(chat.history) == 2
    assert chat.history[0].prompt == prompt
    assert chat.history[0].response == first_response
    assert chat.history[0].prompt_tokens is None
    assert chat.history[0].response_tokens is None
    assert chat.history[0].total_tokens is None
    assert chat.history[0].cost is None
    assert chat.history[1].prompt == second_prompt
    assert chat.history[1].response == second_response
    assert chat.history[1].prompt_tokens is None
    assert chat.history[1].response_tokens is None
    assert chat.history[1].total_tokens is None
    assert chat.history[1].cost is None

    assert chat.total_prompt_tokens is None
    assert chat.total_response_tokens is None
    assert chat.total_tokens is None
    assert chat.total_cost is None

    assert chain.total_tokens is None
    assert chain.total_cost is None

def test_Chain_with_MockChat_tokens_costs():  # noqa
    prompt = "Here's a question."
    first_response = "Response: " + prompt
    second_prompt = "Question: " + first_response
    second_response = "Response: " + second_prompt

    cost_per_token = 15
    # this Chat returns the "Response: " + prompt
    chat = MockChat(
        return_prompt="Response: ",
        token_counter=len,
        cost_per_token=cost_per_token,
    )
    chain = Chain(chain=[chat, lambda x: "Question: " + x, chat])
    result = chain(prompt)
    # the final result should be the response returned by the second invokation of chat()
    assert result == second_response

    # check that the prompts/responses got propegated through the chain
    assert len(chat.history) == 2
    assert chat.history[0].prompt == prompt
    assert chat.history[0].response == first_response
    assert chat.history[0].prompt_tokens == len(prompt)
    assert chat.history[0].response_tokens == len(first_response)
    assert chat.history[0].total_tokens == len(prompt) + len(first_response)
    assert chat.history[0].cost == chat.history[0].total_tokens * cost_per_token
    assert chat.history[1].prompt == second_prompt
    assert chat.history[1].response == second_response
    assert chat.history[1].prompt_tokens == len(second_prompt)
    assert chat.history[1].response_tokens == len(second_response)
    assert chat.history[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history[1].cost == chat.history[1].total_tokens * cost_per_token

    assert chat.total_prompt_tokens == chat.history[0].prompt_tokens + chat.history[1].prompt_tokens  # noqa
    assert chat.total_response_tokens == chat.history[0].response_tokens + chat.history[1].response_tokens  # noqa
    assert chat.total_tokens == chat.history[0].total_tokens + chat.history[1].total_tokens
    assert chat.total_cost == chat.history[0].cost + chat.history[1].cost

    # because the `chat` model is included twice in the chain; this check ensures we are not
    # double-counting the totals
    assert chain.total_tokens == chat.total_tokens
    assert chain.total_cost == chat.total_cost

    new_prompt = "New Prompt"
    result = chain(new_prompt)
    new_first_response = "Response: " + new_prompt
    new_second_prompt = "Question: " + new_first_response
    new_second_response = "Response: " + new_second_prompt
    # the final result should be the response returned by the second invokation of chat()
    assert result == new_second_response

    assert len(chat.history) == 4
    assert chat.history[0].prompt == prompt
    assert chat.history[0].response == first_response
    assert chat.history[0].prompt_tokens == len(prompt)
    assert chat.history[0].response_tokens == len(first_response)
    assert chat.history[0].total_tokens == len(prompt) + len(first_response)
    assert chat.history[0].cost == chat.history[0].total_tokens * cost_per_token
    assert chat.history[1].prompt == second_prompt
    assert chat.history[1].response == second_response
    assert chat.history[1].prompt_tokens == len(second_prompt)
    assert chat.history[1].response_tokens == len(second_response)
    assert chat.history[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history[1].cost == chat.history[1].total_tokens * cost_per_token
    assert chat.history[2].prompt == new_prompt
    assert chat.history[2].response == new_first_response
    assert chat.history[2].prompt_tokens == len(new_prompt)
    assert chat.history[2].response_tokens == len(new_first_response)
    assert chat.history[2].total_tokens == len(new_prompt) + len(new_first_response)
    assert chat.history[2].cost == chat.history[2].total_tokens * cost_per_token
    assert chat.history[3].prompt == new_second_prompt
    assert chat.history[3].response == new_second_response
    assert chat.history[3].prompt_tokens == len(new_second_prompt)
    assert chat.history[3].response_tokens == len(new_second_response)
    assert chat.history[3].total_tokens == len(new_second_prompt) + len(new_second_response)
    assert chat.history[3].cost == chat.history[3].total_tokens * cost_per_token

    assert chat.total_prompt_tokens == chat.history[0].prompt_tokens + \
        chat.history[1].prompt_tokens + \
        chat.history[2].prompt_tokens + \
        chat.history[3].prompt_tokens
    assert chat.total_response_tokens == chat.history[0].response_tokens + \
        chat.history[1].response_tokens + \
        chat.history[2].response_tokens + \
        chat.history[3].response_tokens
    assert chat.total_tokens == chat.history[0].total_tokens + \
        chat.history[1].total_tokens + \
        chat.history[2].total_tokens + \
        chat.history[3].total_tokens
    assert chat.total_cost == chat.history[0].cost + \
        chat.history[1].cost + \
        chat.history[2].cost + \
        chat.history[3].cost

    # because the `chat` model is included twice in the chain; this check ensures we are not
    # double-counting the totals
    assert chain.total_tokens == chat.total_tokens
    assert chain.total_cost == chat.total_cost

def test_Chain_with_MockChat_MockEmbeddings():  # noqa
    """
    This is an unrealistic but useful test where we are using an embeddings model and a chat
    model in a chain multiple times; helper functions are used to change the output of the
    embeddings model to the input of the chat model and vice versa; this is demonstrating/testing
    the extensibility of the chain.
    """  # noqa: D404
    cost_per_token_chat = 15
    cost_per_token_embedding = 7
    # this Chat returns the "Response: " + prompt
    docs = [
        Document(content="Doc A"),
        Document(content="Doc B"),
    ]

    def list_docs_to_prompt(docs: list[Document]) -> str:
        """
        This isn't an example of how a chain would actually be used, but simply a way to mimic
        fowarding information from one step to another.
        """  # noqa: D404
        return ' '.join([x.content for x in docs])

    def prompt_to_list_docs(prompt: str) -> list[Document]:
        """
        This isn't an example of how a chain would actually be used, but simply a way to mimic
        fowarding information from one step to another.
        """  # noqa: D404
        return [Document(content=x) for x in prompt.split(' ')]

    embeddings = MockEmbeddings(token_counter=len, cost_per_token=cost_per_token_embedding)
    chat = MockChat(
        return_prompt="Response: ",
        token_counter=len,
        cost_per_token=cost_per_token_chat,
    )
    chain = Chain(chain=[
        embeddings,
        list_docs_to_prompt,
        chat,
        prompt_to_list_docs,
        embeddings,
        list_docs_to_prompt,
        chat])
    result = chain(docs)

    ####
    # Test chat model
    ####
    initial_prompt = "Doc A Doc B"
    first_response = "Response: " + initial_prompt
    second_prompt = first_response
    second_response = "Response: " + second_prompt

    # the final result should be the response returned by the second invokation of chat()
    assert result == second_response
    # check that the prompts/responses got propegated through the chain
    assert len(chat.history) == 2
    assert chat.history[0].prompt == initial_prompt
    assert chat.history[0].response == first_response
    assert chat.history[0].prompt_tokens == len(initial_prompt)
    assert chat.history[0].response_tokens == len(first_response)
    assert chat.history[0].total_tokens == len(initial_prompt) + len(first_response)
    assert chat.history[0].cost == chat.history[0].total_tokens * cost_per_token_chat
    assert chat.history[1].prompt == second_prompt
    assert chat.history[1].response == second_response
    assert chat.history[1].prompt_tokens == len(second_prompt)
    assert chat.history[1].response_tokens == len(second_response)
    assert chat.history[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history[1].cost == chat.history[1].total_tokens * cost_per_token_chat

    assert chat.total_prompt_tokens == chat.history[0].prompt_tokens + chat.history[1].prompt_tokens  # noqa
    assert chat.total_response_tokens == chat.history[0].response_tokens + chat.history[1].response_tokens  # noqa
    assert chat.total_tokens == chat.history[0].total_tokens + chat.history[1].total_tokens
    assert chat.total_cost == chat.history[0].cost + chat.history[1].cost

    ####
    # Test embeddings model
    ####
    assert len(embeddings.history) == 2
    assert embeddings.history[0].total_tokens == len(docs[0].content) + len(docs[1].content)
    assert embeddings.history[0].cost == embeddings.history[0].total_tokens * cost_per_token_embedding  # noqa
    assert embeddings.history[1].total_tokens == len("Response:DocADocB")
    assert embeddings.history[1].cost == embeddings.history[1].total_tokens * cost_per_token_embedding  # noqa

    assert embeddings.total_tokens == embeddings.history[0].total_tokens + embeddings.history[1].total_tokens  # noqa
    assert embeddings.total_cost == embeddings.history[0].cost + embeddings.history[1].cost

    ####
    # Test chain
    ####
    # because the `chat` model is included twice in the chain; this check ensures we are not
    # double-counting the totals
    assert chain.total_tokens == chat.total_tokens + embeddings.total_tokens
    assert chain.total_cost == chat.total_cost + embeddings.total_cost

    ####
    # Test using the same chain again
    ####
    new_docs = [
        Document(content="Doc CC"),
        Document(content="Doc DD"),
    ]
    new_result = chain(new_docs)
    ####
    # Test chat model
    ####
    new_initial_prompt = "Doc CC Doc DD"
    new_first_response = "Response: " + new_initial_prompt
    new_second_prompt = new_first_response
    new_second_response = "Response: " + new_second_prompt

    # the final result should be the response returned by the second invokation of chat()
    assert new_result == new_second_response
    # check that the prompts/responses got propegated through the chain
    assert len(chat.history) == 4
    # these should not have changed from last time
    assert chat.history[0].prompt == initial_prompt
    assert chat.history[0].response == first_response
    assert chat.history[0].prompt_tokens == len(initial_prompt)
    assert chat.history[0].response_tokens == len(first_response)
    assert chat.history[0].total_tokens == len(initial_prompt) + len(first_response)
    assert chat.history[0].cost == chat.history[0].total_tokens * cost_per_token_chat
    assert chat.history[1].prompt == second_prompt
    assert chat.history[1].response == second_response
    assert chat.history[1].prompt_tokens == len(second_prompt)
    assert chat.history[1].response_tokens == len(second_response)
    assert chat.history[1].total_tokens == len(second_prompt) + len(second_response)
    assert chat.history[1].cost == chat.history[1].total_tokens * cost_per_token_chat
    # test the new history
    assert chat.history[2].prompt == new_initial_prompt
    assert chat.history[2].response == new_first_response
    assert chat.history[2].prompt_tokens == len(new_initial_prompt)
    assert chat.history[2].response_tokens == len(new_first_response)
    assert chat.history[2].total_tokens == len(new_initial_prompt) + len(new_first_response)
    assert chat.history[2].cost == chat.history[2].total_tokens * cost_per_token_chat
    assert chat.history[3].prompt == new_second_prompt
    assert chat.history[3].response == new_second_response
    assert chat.history[3].prompt_tokens == len(new_second_prompt)
    assert chat.history[3].response_tokens == len(new_second_response)
    assert chat.history[3].total_tokens == len(new_second_prompt) + len(new_second_response)
    assert chat.history[3].cost == chat.history[3].total_tokens * cost_per_token_chat

    # test chat totals
    assert chat.total_prompt_tokens == chat.history[0].prompt_tokens + \
        chat.history[1].prompt_tokens + \
        chat.history[2].prompt_tokens + \
        chat.history[3].prompt_tokens
    assert chat.total_response_tokens == chat.history[0].response_tokens + \
        chat.history[1].response_tokens + \
        chat.history[2].response_tokens + \
        chat.history[3].response_tokens
    assert chat.total_tokens == chat.history[0].total_tokens + \
        chat.history[1].total_tokens + \
        chat.history[2].total_tokens + \
        chat.history[3].total_tokens
    assert chat.total_cost == chat.history[0].cost + \
        chat.history[1].cost + \
        chat.history[2].cost + \
        chat.history[3].cost

    ####
    # Test embeddings model
    ####
    assert len(embeddings.history) == 4
    # original history should not have changed
    assert embeddings.history[0].total_tokens == len(docs[0].content) + len(docs[1].content)
    assert embeddings.history[0].cost == embeddings.history[0].total_tokens * cost_per_token_embedding  # noqa
    assert embeddings.history[1].total_tokens == len("Response:DocADocB")
    assert embeddings.history[1].cost == embeddings.history[1].total_tokens * cost_per_token_embedding  # noqa
    # test new history
    assert embeddings.history[2].total_tokens == len(new_docs[0].content) + len(new_docs[1].content)  # noqa
    assert embeddings.history[2].cost == embeddings.history[2].total_tokens * cost_per_token_embedding  # noqa
    assert embeddings.history[3].total_tokens == len("Response:DocCCDocDD")
    assert embeddings.history[3].cost == embeddings.history[3].total_tokens * cost_per_token_embedding  # noqa

    assert embeddings.total_tokens == embeddings.history[0].total_tokens + \
        embeddings.history[1].total_tokens + \
        embeddings.history[2].total_tokens + \
        embeddings.history[3].total_tokens
    assert embeddings.total_cost == embeddings.history[0].cost + \
        embeddings.history[1].cost + \
        embeddings.history[2].cost + \
        embeddings.history[3].cost

    ####
    # Test chain
    ####
    # because the `chat` model is included twice in the chain; this check ensures we are not
    # double-counting the totals
    assert chain.total_tokens == chat.total_tokens + embeddings.total_tokens
    assert chain.total_cost == chat.total_cost + embeddings.total_cost
