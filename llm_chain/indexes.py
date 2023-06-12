"""TODO."""
import chromadb
from chromadb.api.models.Collection import Collection
from llm_chain.base import Document, DocumentIndex, EmbeddingsModel

class ChromaDocumentIndex(DocumentIndex):
    """TODO."""

    def __init__(
            self,
            embeddings_model: EmbeddingsModel,
            collection: Collection | None = None) -> None:
        self._collection = collection or chromadb.Client().create_collection('temp')
        self._embeddings_model = embeddings_model

    def add_documents(self, docs: list[Document]) -> None:
        """TODO."""
        embeddings = self._embeddings_model(docs=docs)
        metadatas = [x.metadata or {} for x in docs]
        documents = [x.content for x in docs]
        self._collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=[str(x) for x in range(len(docs))],
        )

    def search_documents(self, doc: Document, n_results: int = 3) -> list[Document]:
        """TODO."""
        embeddings = self._embeddings_model(docs=[doc])
        results = self._collection.query(
            query_embeddings=embeddings,
            n_results=n_results,
        )
        # index 0 because we are only searching against a single document
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        similar_docs = []
        for doc, meta, dist in zip(documents, metadatas, distances, strict=True):
            meta['distance'] = dist
            similar_docs.append(Document(
                content=doc,
                metadata=meta,
            ))
        return similar_docs

    @property
    def total_tokens(self) -> str:
        """Propagates the total_tokens used by the underlying EmbeddingsModel."""
        return self._embeddings_model.total_tokens

    @property
    def total_cost(self) -> str:
        """Propagates the total_cost used by the underlying EmbeddingsModel."""
        return self._embeddings_model.total_cost
