"""tests llm_chain/tools.py."""
from llm_chain.base import Document
from llm_chain.tools import DuckDuckGoSearch, split_documents, scrape_url


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
