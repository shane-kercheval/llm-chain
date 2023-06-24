"""TODO."""
from itertools import islice
from llm_chain.base import Document, HistoricalData, Record


def split_documents(docs: list[Document], max_chunk_size: int = 500) -> list[Document]:
    """TODO."""
    new_docs = []
    for doc in docs:
        if doc.content:
            content = doc.content
            metadata = doc.metadata
            while len(content) > max_chunk_size:
                new_doc = Document(
                    content=content[:max_chunk_size],
                    metadata=metadata,
                )
                new_docs.append(new_doc)
                content = content[max_chunk_size:]  # remove the chunk we added
            # check for remaining/leftover content
            if content:
                new_docs.append(Document(
                    content=content,
                    metadata=metadata,
                ))
        else:
            # edge case for empty document; we need to make sure it is added to the list
            new_docs.append(doc)

    return new_docs


def scrape_url(url: str) -> str:
    """TODO."""
    # TODO: test
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url)
    assert response.status_code == 200
    soup = BeautifulSoup(response.content, 'html.parser')
    return soup.get_text().strip()


class SearchRecord(Record):
    """TODO."""

    query: str
    results: list[dict]


class DuckDuckGoSearch(HistoricalData):
    """TODO."""

    def __init__(self, top_n: int = 3):
        self.top_n = top_n
        self._history = []

    def __call__(self, query: str) -> list[dict]:
        """TODO."""
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            ddgs_generator = ddgs.text(query, region='wt-wt', safesearch='Off', timelimit='y')
            results = list(islice(ddgs_generator, self.top_n))
        self._history.append(SearchRecord(
            query=query,
            results=results,
        ))
        return results

    @property
    def history(self) -> list[SearchRecord]:
        """TODO."""
        return self._history
