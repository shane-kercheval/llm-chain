"""TODO."""
import os
from itertools import islice
import re
from bs4 import BeautifulSoup
import requests
from datetime import datetime
from pydantic import BaseModel, Field, validator
from markdownify import markdownify as html_to_markdown
from llm_chain.utilities import retry_handler
from llm_chain.base import Document, HistoricalData, Record, RequestError


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
    import requests
    from bs4 import BeautifulSoup
    response = requests.get(url)
    if response.status_code != 200:  # let client decide what to do
        raise RequestError(status_code=response.status_code, reason=response.reason)
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


class StackAnswer(BaseModel):
    """TODO."""

    answer_id: int
    is_accepted: bool
    score: int
    body: str
    text: str | None
    markdown: str | None
    creation_date: int

    def __init__(self, **data):  # noqa
        super().__init__(**data)
        self.text = BeautifulSoup(self.body, 'html.parser').get_text(separator=' ')
        self.markdown = html_to_markdown(self.body)


    @validator('creation_date')
    def convert_to_datetime(cls, value: str) -> datetime:  # noqa: N805
        """Convert from string to datetime."""
        return datetime.fromtimestamp(value)


class StackQuestion(BaseModel):
    """TODO."""

    question_id: int
    score: int
    creation_date: int
    answer_count: int
    title: str
    link: str
    body: str
    text: str | None
    markdown: str | None
    content_license: str | None
    answers: list[StackAnswer] = Field(default_factory=list)

    def __init__(self, **data):  # noqa
        super().__init__(**data)
        self.text = BeautifulSoup(self.body, 'html.parser').get_text(separator=' ')
        self.markdown = html_to_markdown(self.body)

    @validator('creation_date')
    def convert_to_datetime(cls, value: str) -> datetime:  # noqa: N805
        """Convert from string to datetime."""
        return datetime.fromtimestamp(value)


def _get_stack_overflow_answers(question_id: int, max_answers: int = 2) -> list[StackAnswer]:
    """For a given question_id on Stack Overflow, returns the top `num_answers`."""
    params = {
        "site": "stackoverflow",
        "key": os.getenv('STACK_OVERFLOW_KEY'),
        "filter": "withbody",  # Include the answer body in the response
        "sort": "votes",  # Sort answers by votes (highest first)
        "pagesize": max_answers,  # Fetch only the top answers
    }
    response = retry_handler()(
        requests.get,
        f"https://api.stackexchange.com/2.3/questions/{question_id}/answers",
        params=params,
    )
    if response.status_code != 200:  # let client decide what to do
        raise RequestError(status_code=response.status_code, reason=response.reason)
    answers = response.json().get('items', [])
    return [StackAnswer(**x) for x in answers]


def search_stack_overflow(
        query: str,
        max_questions: int = 2,
        max_answers: int = 2) -> list[StackQuestion]:
    """TODO."""
    params = {
        'site': 'stackoverflow',
        'key': os.getenv('STACK_OVERFLOW_KEY'),
        'q': query,
        'sort': 'relevance',
        'order': 'desc',
        'filter': 'withbody',  # Include the question body in the response
        'pagesize': max_questions,
        'page': 1,
    }
    response = retry_handler()(
        requests.get,
        'https://api.stackexchange.com/2.3/search/advanced',
        params=params,
    )
    assert response.status_code == 200
    questions = response.json().get('items', [])
    questions = [StackQuestion(**x) for x in questions]

    for question in questions:
        if question.answer_count > 0:
            question.answers = _get_stack_overflow_answers(
                question_id=question.question_id,
                max_answers=max_answers,
            )
    return questions
