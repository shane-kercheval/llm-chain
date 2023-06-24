"""TODO."""
import chromadb
from chromadb.api.models.Collection import Collection
from llm_chain.base import Document, DocumentIndex, EmbeddingsModel, EmbeddingsRecord
from llm_chain.utilities import create_hash


class ChromaDocumentIndex(DocumentIndex):
    """TODO."""

    def __init__(
            self,
            embeddings_model: EmbeddingsModel,
            collection: Collection | None = None,
            n_results: int = 3) -> None:
        super().__init__(n_results=n_results)
        self._collection = collection or chromadb.Client().create_collection('temp')
        self._embeddings_model = embeddings_model

    def add(self, docs: list[Document]) -> None:
        """TODO."""
        if not docs:
            return
        existing_ids = set(self._collection.get(include=['documents'])['ids'])
        ids = []
        metadatas = []
        contents = []
        documents = []
        for doc in docs:
            doc_id = create_hash(doc.content)
            if doc_id not in existing_ids:
                ids.append(doc_id)
                metadatas.append(doc.metadata or {})
                contents.append(doc.content)
                documents.append(doc)

        embeddings = self._embeddings_model(docs=documents)
        if documents:
            self._collection.add(
                embeddings=embeddings,
                metadatas=metadatas,
                documents=contents,
                ids=ids,
            )

    def _search(self, doc: Document | str, n_results: int) -> list[Document]:
        embeddings = self._embeddings_model(docs=doc)
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
    def history(self) -> list[EmbeddingsRecord]:
        """TODO."""
        return self._embeddings_model.history
