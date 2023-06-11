"""test llm_chain/vector_db/chroma.db."""
import chromadb
from llm_chain.base import Document
from llm_chain.vector_db.chroma import ChromaCollection


def test_chroma_add_search_documents():  # noqa
    client = chromadb.Client()
    collection = client.create_collection("test")
    chroma_db = ChromaCollection(collection=collection)

    docs = [
        Document(content="Doc A", metadata={'id': 0}, embedding=[0.5, 0.5, 0.5, 0.5, 0.5]),
        Document(content="Doc B", metadata={'id': 1}, embedding=[1, 1, 1, 1, 1]),
        Document(content="Doc C", metadata={'id': 3}, embedding=[3, 3, 3, 3, 3]),
        Document(content="Doc D", metadata={'id': 4}, embedding=[4, 4, 4, 4, 4]),
    ]
    chroma_db.add_documents(docs=docs)
    # verify documents where added to collection
    collection_docs = collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs]
    assert collection_docs['metadatas'] == [x.metadata for x in docs]
    assert collection_docs['ids'] == ['0', '1', '2', '3']
    assert collection_docs['embeddings'] == [x.embedding for x in docs]

    results = chroma_db.search_documents(doc=docs[1], n_results=3)
    assert len(results) == 3
    # first/best result should match doc 1
    assert results[0].content == docs[1].content
    assert results[0].metadata['id'] == docs[1].metadata['id']
    assert results[0].metadata['distance'] == 0
    # second result should match doc 0
    assert results[1].content == docs[0].content
    assert results[1].metadata['id'] == docs[0].metadata['id']
    assert results[1].metadata['distance'] > 0
    # third result should match doc 3 (index 2)
    assert results[2].content == docs[2].content
    assert results[2].metadata['id'] == docs[2].metadata['id']
    assert results[2].metadata['distance'] > results[1].metadata['distance']

    results = chroma_db.search_documents(doc=docs[1], n_results=1)
    assert len(results) == 1
    # first/best result should match doc 1
    assert results[0].content == docs[1].content
    assert results[0].metadata['id'] == docs[1].metadata['id']
    assert results[0].metadata['distance'] == 0
