"""tests llm_chain/tools.py."""
import os
import pytest
import requests
from llm_chain.base import Document, RequestError
from llm_chain.tools import DuckDuckGoSearch, _get_stack_overflow_answers, search_stack_overflow, \
    split_documents, scrape_url


def test_split_documents__preserve_words_false():  # noqa
    max_chunk_size = 10

    result = split_documents([], max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content=' ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content='  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    docs = [
        Document(content='\n', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == []

    # test that space does not affect result
    docs = [
        Document(content='0123 5678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == docs

    # test that leading and trailing space gets stripped
    docs = [
        Document(content=' 123 567 ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    docs[0].content = docs[0].content.strip()
    assert result == docs


    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    docs = [
        Document(content='012345678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == docs

    docs = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    assert result == docs

    docs = [
        Document(content='0123456789a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=False)
    expected_result = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='0123 5678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123 56789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123 56789a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='0123 56789abcdefghi', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='0123 56789abcdefghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123 56789abcdefghijk', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    expected_result = [
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='0123 5678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123 56789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123 56789', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='0123 56789', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='abcdefghi', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='0123 56789', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='abcdefghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123 56789', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='abcdefghij', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='k', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    result = split_documents(docs, max_chunk_size, preserve_words=False)
    assert result == expected_result

def test_split_documents__preserve_words_true():  # noqa
    max_chunk_size = 10

    result = split_documents([], max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content=' ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content='  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    docs = [
        Document(content='\n', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == []

    # test that space does not affect result
    docs = [
        Document(content='0123 5678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == docs

    # test that leading and trailing space gets stripped
    docs = [
        Document(content=' 123 567 ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    # test with 10 characters
    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    # test with 11 characters
    docs = [
        Document(content=' 123 567  ', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    docs[0].content = docs[0].content.strip()
    assert result == docs

    # test with 9 characaters
    docs = [
        Document(content='012345678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == docs

    # test with 10 characters no space
    docs = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    assert result == docs

    # test with 11 characters and no space; since no space was found, it splits up the word
    docs = [
        Document(content='0123456789a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123456789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 567 9a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 567', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='9a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 56789\nabc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 56789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='abc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 56789\n abc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 56789', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='abc', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    # test with 11 characters and space; it should preserve whole words
    docs = [
        Document(content='0123 5678\n 1234 67890 a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='0123 5678', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='1234 67890', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    docs = [
        Document(content='This is a normal sentence.', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),  # noqa
    ]
    result = split_documents(docs, max_chars=max_chunk_size, preserve_words=True)
    expected_result = [
        Document(content='This is a', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='normal', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='sentence.', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
    ]
    assert result == expected_result

    docs = [
        Document(content='', metadata={'doc': 0}, embedding=[0, 0, 0, 0]),
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='012345678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123456789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123456789a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content=' 0123456789ab\ndefghijk', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content=' 0 23456789abcd fghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123456789abcdefghijk ', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    expected_result = [
        Document(content='0', metadata={'doc': 1}, embedding=[1, 1, 1, 1]),
        Document(content='012345678', metadata={'doc': 2}, embedding=[2, 2, 2, 2]),
        Document(content='0123456789', metadata={'doc': 3}, embedding=[3, 3, 3, 3]),
        Document(content='0123456789', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='a', metadata={'doc': 4}, embedding=[4, 4, 4, 4]),
        Document(content='0123456789', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='ab', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='defghijk', metadata={'doc': 5}, embedding=[5, 5, 5, 5]),
        Document(content='0', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='23456789ab', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='cd fghij', metadata={'doc': 6}, embedding=[6, 6, 6, 6]),
        Document(content='0123456789', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='abcdefghij', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
        Document(content='k', metadata={'doc': 7}, embedding=[7, 7, 7, 7]),
    ]
    result = split_documents(docs, max_chunk_size, preserve_words=True)
    assert result == expected_result

def test_DuckDuckGoSearch():  # noqa
    query = "What is an agent in langchain?"
    search = DuckDuckGoSearch(top_n=1)
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history) == 1
    assert search.history[0].query == query
    assert search.history[0].results == results

    query = "What is langchain?"
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history) == 2
    assert search.history[1].query == query
    assert search.history[1].results == results

def test_scrape_url():  # noqa
    text = scrape_url(url='https://example.com/')
    assert 'example' in text.lower()

def test_scrape_url_404():  # noqa
    with pytest.raises(RequestError):
         scrape_url(url="https://example.com/asdf")

def test_search_stack_overflow():  # noqa
    # not sure how to test this in a way that won't break if the response from stack overflow
    # changes in the future
    # TODO: I don't want to make the tests fail when running on github workflows or someone is
    # building locally; but approach this will silently skip tests which is not ideal
    if os.getenv('STACK_OVERFLOW_KEY', None):
        # this question gets over 25K upvotes and has many answers; let's make sure we get the
        # expected number of questions/answers
        question = "Why is processing a sorted array faster than processing an unsorted array?"
        results = search_stack_overflow(query=question, max_questions=1, max_answers=1)
        assert results
        assert len(results) == 1
        assert results[0].title == question
        assert results[0].answer_count > 1
        assert len(results[0].answers) == 1
        # check that the body of the question contains html but the text/markdown does not
        assert '<p>' in results[0].body
        assert len(results[0].body) > 100
        assert '<p>' not in results[0].text
        assert len(results[0].text) > 100
        assert '<p>' not in results[0].markdown
        assert len(results[0].markdown) > 100
        # check that the body of the answer contains html but the text/markdown does not
        assert '<p>' in results[0].answers[0].body
        assert len(results[0].answers[0].body) > 100
        assert '<p>' not in results[0].answers[0].text
        assert len(results[0].answers[0].text) > 100
        assert '<p>' not in results[0].answers[0].markdown
        assert len(results[0].answers[0].markdown) > 100

        question = "getting segmentation fault in linux"
        results = search_stack_overflow(query=question, max_questions=2, max_answers=2)
        assert results
        assert len(results) > 1
        assert any(x for x in results if x.answer_count > 0)

        # make sure the function doesn't fail when there are no matches/results
        assert search_stack_overflow(query="asdfasdfasdfasdflkasdfljsadlkfjasdlkfja") == []

def test_RequestError():  # noqa
    response = requests.get("https://example.com/asdf")
    assert RequestError(status_code=response.status_code, reason=response.reason)

def test__get_stack_overflow_answers_404():  # noqa
     if os.getenv('STACK_OVERFLOW_KEY', None):
        with pytest.raises(RequestError):
            _ = _get_stack_overflow_answers(question_id='asdf')
