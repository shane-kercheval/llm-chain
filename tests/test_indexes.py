"""test llm_chain/vector_db/chroma.db."""
import chromadb
from llm_chain.base import Document
from llm_chain.indexes import ChromaDocumentIndex
from tests.conftest import MockABCDEmbeddings, MockRandomEmbeddings


def test_chroma_add_search_documents(fake_docs_abcd):  # noqa

    embeddings_model = MockABCDEmbeddings()
    client = chromadb.Client()
    collection = client.create_collection("test")
    chroma_db = ChromaDocumentIndex(collection=collection, embeddings_model=embeddings_model)
    assert chroma_db.total_tokens is None
    assert chroma_db.total_cost is None
    chroma_db.add_documents(docs=fake_docs_abcd)

    initial_expected_tokens = len("Doc X") * len(fake_docs_abcd)
    initial_expected_cost = initial_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == initial_expected_tokens
    assert chroma_db.total_cost == initial_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.total_cost == embeddings_model.total_cost
    assert len(embeddings_model.history) == 1
    assert embeddings_model.history[0].total_tokens == initial_expected_tokens
    assert embeddings_model.history[0].cost == initial_expected_cost

    # verify documents and embeddings where added to collection
    collection_docs = collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in fake_docs_abcd]
    assert collection_docs['metadatas'] == [x.metadata for x in fake_docs_abcd]
    assert collection_docs['ids'] == ['0', '1', '2', '3']
    assert collection_docs['embeddings'] == list(embeddings_model.lookup.values())

    # search based on first doc
    results = chroma_db.search_documents(doc=fake_docs_abcd[1], n_results=3)
    assert len(results) == 3
    # first/best result should match doc 1
    assert results[0].content == fake_docs_abcd[1].content
    assert results[0].metadata['id'] == fake_docs_abcd[1].metadata['id']
    assert results[0].metadata['distance'] == 0
    # second result should match doc 0
    assert results[1].content == fake_docs_abcd[0].content
    assert results[1].metadata['id'] == fake_docs_abcd[0].metadata['id']
    assert results[1].metadata['distance'] > 0
    # third result should match doc 3 (index 2)
    assert results[2].content == fake_docs_abcd[2].content
    assert results[2].metadata['id'] == fake_docs_abcd[2].metadata['id']
    assert results[2].metadata['distance'] > results[1].metadata['distance']

    new_expected_tokens = initial_expected_tokens + len('Doc X')
    new_expected_cost = new_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == new_expected_tokens
    assert chroma_db.total_cost == new_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.total_cost == embeddings_model.total_cost
    assert len(embeddings_model.history) == 2
    assert embeddings_model.history[0].total_tokens == initial_expected_tokens
    assert embeddings_model.history[0].cost == initial_expected_cost
    assert embeddings_model.history[1].total_tokens == len("Doc X")
    assert embeddings_model.history[1].cost == len("Doc X") * embeddings_model.cost_per_token

    # search based on third doc
    results = chroma_db.search_documents(doc=fake_docs_abcd[2], n_results=1)
    assert len(results) == 1
    # first/best result should match doc 2
    assert results[0].content == fake_docs_abcd[2].content
    assert results[0].metadata['id'] == fake_docs_abcd[2].metadata['id']
    assert results[0].metadata['distance'] == 0

    new_expected_tokens += len('Doc X')
    new_expected_cost = new_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == new_expected_tokens
    assert chroma_db.total_cost == new_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.total_cost == embeddings_model.total_cost
    assert len(embeddings_model.history) == 3
    assert embeddings_model.history[0].total_tokens == initial_expected_tokens
    assert embeddings_model.history[0].cost == initial_expected_cost
    assert embeddings_model.history[1].total_tokens == len("Doc X")
    assert embeddings_model.history[1].cost == len("Doc X") * embeddings_model.cost_per_token
    assert embeddings_model.history[1].total_tokens == len("Doc X")
    assert embeddings_model.history[1].cost == len("Doc X") * embeddings_model.cost_per_token

def test_chroma_add_document_without_metadata():  # noqa
    cost_per_token = 13
    embeddings_model = MockRandomEmbeddings(token_counter=len, cost_per_token=cost_per_token)
    # test without passing a collection
    doc_index = ChromaDocumentIndex(embeddings_model=embeddings_model)
    docs = [
        Document(content='This is a document'),
        Document(content='This is a another document'),
        Document(content='This is a another another document'),
    ]
    doc_index.add_documents(docs=docs)
    collection_docs = doc_index._collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs]
    assert collection_docs['metadatas'] == [{}, {}, {}]
    assert collection_docs['ids'] == ['0', '1', '2']
    assert len(collection_docs['embeddings']) == len(docs)

    results = doc_index.search_documents(doc=docs[0], n_results=1)
    assert 'distance' in results[0].metadata
