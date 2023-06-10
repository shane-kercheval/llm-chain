"""TODO."""
from llm_chain.base import Document


def split_documents(docs: list[Document], max_chunk_size: int = 500) -> list[Document]:
    """TODO."""
    new_docs = []
    for doc in docs:
        if doc.content:
            content = doc.content
            metadata = doc.metadata
            embedding = doc.embedding
            while len(content) > max_chunk_size:
                new_doc = Document(
                    content=content[:max_chunk_size],
                    metadata=metadata,
                    embedding=embedding,
                )
                new_docs.append(new_doc)
                content = content[max_chunk_size:]  # remove the chunk we added
            # check for remaining/leftover content
            if content:
                new_docs.append(Document(
                    content=content,
                    metadata=metadata,
                    embedding=embedding,
                ))
        else:
            # edge case for empty document; we need to make sure it is added to the list
            new_docs.append(doc)

    return new_docs
