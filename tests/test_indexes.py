"""test llm_chain/vector_db/chroma.db."""
import chromadb
from llm_chain.base import Document, DocumentIndex, Record
from llm_chain.indexes import ChromaDocumentIndex
from tests.conftest import MockABCDEmbeddings, MockRandomEmbeddings


class MockIndex(DocumentIndex):  # noqa
    def __init__(self) -> None:
        super().__init__()
        self.documents = []

    def add(self, docs: list[Document]) -> None:  # noqa
        self.documents += docs

    def search(self, doc: Document, n_results: int = 3) -> list[Document]:  # noqa
        return self.documents

    @property
    def history(self) -> list[Record]:  # noqa
        return [Record()]


def test_base_index():  # noqa
    mock_index = MockIndex()
    # test `call()` when passing list[Document] which should call `add_documents()`
    documents_to_add = [Document(content='Doc A'), Document(content='Doc A')]
    return_value = mock_index(documents_to_add)
    assert return_value is None
    assert mock_index.documents == documents_to_add
    # test `call()` when passing Document which should call `search_documents()`
    return_value = mock_index(Document(content='doc'))
    assert return_value == documents_to_add

def test_chroma_add_search_documents(fake_docs_abcd):  # noqa
    embeddings_model = MockABCDEmbeddings()
    client = chromadb.Client()
    collection = client.create_collection("test")
    chroma_db = ChromaDocumentIndex(collection=collection, embeddings_model=embeddings_model)
    assert chroma_db.total_tokens is None
    assert chroma_db.cost is None
    chroma_db.add(docs=None)
    assert chroma_db.total_tokens is None
    assert chroma_db.cost is None
    chroma_db.add(docs=[])
    assert chroma_db.total_tokens is None
    assert chroma_db.cost is None

    chroma_db.add(docs=fake_docs_abcd)
    initial_expected_tokens = len("Doc X") * len(fake_docs_abcd)
    initial_expected_cost = initial_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == initial_expected_tokens
    assert chroma_db.cost == initial_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.cost == embeddings_model.cost
    assert len(embeddings_model._history) == 1
    assert embeddings_model._history[0].total_tokens == initial_expected_tokens
    assert embeddings_model._history[0].cost == initial_expected_cost

    # verify documents and embeddings where added to collection
    collection_docs = collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in fake_docs_abcd]
    assert collection_docs['metadatas'] == [x.metadata for x in fake_docs_abcd]
    assert len(collection_docs['ids']) == 4
    assert collection_docs['embeddings'] == list(embeddings_model.lookup.values())

    # search based on first doc
    results = chroma_db.search(doc=fake_docs_abcd[1], n_results=3)
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
    assert chroma_db.cost == new_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.cost == embeddings_model.cost
    assert len(embeddings_model._history) == 2
    assert embeddings_model._history[0].total_tokens == initial_expected_tokens
    assert embeddings_model._history[0].cost == initial_expected_cost
    assert embeddings_model._history[1].total_tokens == len("Doc X")
    assert embeddings_model._history[1].cost == len("Doc X") * embeddings_model.cost_per_token

    # search based on third doc
    results = chroma_db.search(doc=fake_docs_abcd[2], n_results=1)
    assert len(results) == 1
    # first/best result should match doc 2
    assert results[0].content == fake_docs_abcd[2].content
    assert results[0].metadata['id'] == fake_docs_abcd[2].metadata['id']
    assert results[0].metadata['distance'] == 0

    new_expected_tokens += len('Doc X')
    new_expected_cost = new_expected_tokens * embeddings_model.cost_per_token
    # test that usage reflects underlying usage in embeddings
    assert chroma_db.total_tokens == new_expected_tokens
    assert chroma_db.cost == new_expected_cost
    assert chroma_db.total_tokens == embeddings_model.total_tokens
    assert chroma_db.cost == embeddings_model.cost
    assert len(embeddings_model._history) == 3
    assert embeddings_model._history[0].total_tokens == initial_expected_tokens
    assert embeddings_model._history[0].cost == initial_expected_cost
    assert embeddings_model._history[1].total_tokens == len("Doc X")
    assert embeddings_model._history[1].cost == len("Doc X") * embeddings_model.cost_per_token
    assert embeddings_model._history[1].total_tokens == len("Doc X")
    assert embeddings_model._history[1].cost == len("Doc X") * embeddings_model.cost_per_token

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
    doc_index.add(docs=docs)
    collection_docs = doc_index._collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs]
    assert collection_docs['metadatas'] == [{}, {}, {}]
    assert len(collection_docs['ids']) == len(docs)
    assert len(collection_docs['embeddings']) == len(docs)

    results = doc_index.search(doc=docs[0], n_results=1)
    assert 'distance' in results[0].metadata

    # test adding same documents
    new_docs = [
        Document(content='New Doc 1', metadata={'id': 0}),
        Document(content='New Doc 2', metadata={'id': 1}),
    ]
    doc_index.add(docs=docs + new_docs)
    collection_docs = doc_index._collection.get(include = ['documents', 'metadatas', 'embeddings'])
    assert collection_docs['documents'] == [x.content for x in docs + new_docs]
    assert collection_docs['metadatas'] == [{}, {}, {}, {'id': 0}, {'id': 1}]
    assert len(collection_docs['ids']) == len(docs + new_docs)
    assert len(collection_docs['embeddings']) == len(docs + new_docs)

    results = doc_index.search(doc=docs[0], n_results=1)
    assert 'distance' in results[0].metadata
