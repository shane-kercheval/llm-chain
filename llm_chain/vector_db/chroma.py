"""TODO."""
from chromadb.api.models.Collection import Collection
from llm_chain.base import Document

class ChromaCollection:
    """TODO."""

    def __init__(self, collection: Collection) -> None:
        self.collection = collection

    def add_documents(self, docs: list[Document]) -> None:
        """TODO."""
        embeddings = [x.embedding for x in docs]
        metadatas = [x.metadata for x in docs]
        documents = [x.content for x in docs]
        self.collection.add(
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
            ids=[str(x) for x in range(len(docs))],
        )

    def search_documents(self, doc: Document, n_results: int = 3) -> list[Document]:
        """TODO."""
        results = self.collection.query(
            query_embeddings=[doc.embedding],
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
