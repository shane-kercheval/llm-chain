"""TODO."""
from itertools import islice
import re
from llm_chain.base import Document, HistoricalData, Record


def split_documents(
        docs: list[Document],
        max_chars: int = 500,
        preserve_words: bool = True) -> list[Document]:
    """TODO. Mention that it keeps the entire word. Does not return empty documents."""
    new_docs = []
    for doc in docs:
        if doc.content:
            content = doc.content.strip()
            metadata = doc.metadata
            while len(content) > max_chars:
                # find the last space that is within the limit
                if preserve_words:
                    split_at = max_chars
                    # find the last whitespace that is within the limit
                    # if no whitespace found, take the whole chunk
                    # we are going 1 beyond the limit in case it is whitespace so that
                    # we can keep everything up until that point
                    for match in re.finditer(r'\s', content[:max_chars + 1]):
                        split_at = match.start()
                else:
                    split_at = max_chars
                new_doc = Document(
                    content=content[:split_at].strip(),
                    metadata=metadata,
                )
                new_docs.append(new_doc)
                content = content[split_at:].strip()  # remove the chunk we added
            # check for remaining/leftover content
            if content:
                new_docs.append(Document(
                    content=content,
                    metadata=metadata,
                ))
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
