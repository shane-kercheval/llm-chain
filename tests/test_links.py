"""Test Link."""
from llm_chain.base import EmbeddingRecord, ExchangeRecord, Link, Record, UsageRecord
from llm_chain.links import DuckDuckGoSearch, SearchRecord


class MockLink(Link):
    """Mocks a Link object."""

    def __init__(self) -> None:
        self._history = []

    def __call__(self, record: Record) -> None:
        """Adds the record to the history."""
        return self._history.append(record)

    @property
    def history(self) -> list[Record]:
        """Returns history."""
        return self._history


def test_history_tracker():  # noqa
    tracker = MockLink()
    assert tracker.history == tracker.history_filter()
    assert tracker.history == []
    assert tracker.history_filter(Record) == []
    assert tracker.history_filter(UsageRecord) == []
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 0
    assert tracker.calculate_historical(name='total_tokens') == 0
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_a = Record(metadata={'id': 'a'})
    tracker(record_a)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a]
    assert tracker.history_filter(Record) == [record_a]
    assert tracker.history_filter(UsageRecord) == []
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 0
    assert tracker.calculate_historical(name='total_tokens') == 0
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_b = UsageRecord(total_tokens=1, metadata={'id': 'b'})
    tracker(record_b)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b]
    assert tracker.history_filter(Record) == [record_a, record_b]
    assert tracker.history_filter(UsageRecord) == [record_b]
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 0
    assert tracker.calculate_historical(name='total_tokens') == 1
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_c = UsageRecord(cost=3, metadata={'id': 'c'})
    tracker(record_c)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c]
    assert tracker.history_filter(UsageRecord) == [record_b, record_c]
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 3
    assert tracker.calculate_historical(name='total_tokens') == 1
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_d = UsageRecord(total_tokens=7, cost=6, metadata={'id': 'd'})
    tracker(record_d)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d]
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d]
    assert tracker.history_filter(ExchangeRecord) == []
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 9
    assert tracker.calculate_historical(name='total_tokens') == 8
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0

    record_e = ExchangeRecord(
        prompt="the prompt",
        response="the response",
        cost=20,
        total_tokens=10,
        metadata={'id': 'e'},
    )
    tracker(record_e)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d, record_e]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d, record_e]
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d, record_e]
    assert tracker.history_filter(ExchangeRecord) == [record_e]
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 29
    assert tracker.calculate_historical(name='total_tokens') == 18
    assert tracker.calculate_historical(name='prompt_tokens') == 0
    assert tracker.calculate_historical(name='response_tokens') == 0
    # test calculating historical values based on ExchangeRecords
    assert tracker.calculate_historical(name='cost', record_types=ExchangeRecord) == 20
    assert tracker.calculate_historical(name='total_tokens', record_types=ExchangeRecord) == 10
    assert tracker.calculate_historical(name='prompt_tokens', record_types=ExchangeRecord) == 0
    assert tracker.calculate_historical(name='response_tokens', record_types=ExchangeRecord) == 0

    record_f = ExchangeRecord(
        prompt="the prompt",
        response="the response",
        cost=13,
        total_tokens=5,
        prompt_tokens=7,
        response_tokens=9,
        metadata={'id': 'f'},
    )
    tracker(record_f)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d, record_e, record_f]
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d, record_e, record_f]  # noqa
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d, record_e, record_f]  # noqa
    assert tracker.history_filter(ExchangeRecord) == [record_e, record_f]
    assert tracker.history_filter(EmbeddingRecord) == []
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 42
    assert tracker.calculate_historical(name='total_tokens') == 23
    assert tracker.calculate_historical(name='prompt_tokens') == 7
    assert tracker.calculate_historical(name='response_tokens') == 9
    # test calculating historical values based on ExchangeRecords
    assert tracker.calculate_historical(name='cost', record_types=ExchangeRecord) == 33
    assert tracker.calculate_historical(name='total_tokens', record_types=ExchangeRecord) == 15
    assert tracker.calculate_historical(name='prompt_tokens', record_types=ExchangeRecord) == 7
    assert tracker.calculate_historical(name='response_tokens', record_types=ExchangeRecord) == 9

    record_g = EmbeddingRecord(
        cost=1,
        total_tokens=2,
        metadata={'id': 'g'},
    )
    tracker(record_g)
    assert tracker.history == tracker.history_filter()
    assert tracker.history == [record_a, record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history_filter(Record) == [record_a, record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history_filter(UsageRecord) == [record_b, record_c, record_d, record_e, record_f, record_g]  # noqa
    assert tracker.history_filter(ExchangeRecord) == [record_e, record_f]
    assert tracker.history_filter(EmbeddingRecord) == [record_g]
    assert tracker.calculate_historical(name='does_not_exist') == 0
    assert tracker.calculate_historical(name='cost') == 43
    assert tracker.calculate_historical(name='total_tokens') == 25
    assert tracker.calculate_historical(name='prompt_tokens') == 7
    assert tracker.calculate_historical(name='response_tokens') == 9
    # test calculating historical values based on EmbeddingRecords
    assert tracker.calculate_historical(name='cost', record_types=EmbeddingRecord) == 1
    assert tracker.calculate_historical(name='total_tokens', record_types=EmbeddingRecord) == 2
    assert tracker.calculate_historical(name='prompt_tokens', record_types=EmbeddingRecord) == 0
    assert tracker.calculate_historical(name='response_tokens', record_types=EmbeddingRecord) == 0
    # test calculating historical values based on ExchangeRecords or EmbeddingRecord
    assert tracker.calculate_historical(
            name='cost',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 34
    assert tracker.calculate_historical(
            name='total_tokens',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 17
    assert tracker.calculate_historical(
            name='prompt_tokens',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 7
    assert tracker.calculate_historical(
            name='response_tokens',
            record_types=(ExchangeRecord, EmbeddingRecord),
        ) == 9

def test_DuckDuckGoSearch():  # noqa
    query = "What is an agent in langchain?"
    search = DuckDuckGoSearch(top_n=1)
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history) == 1
    assert search.history[0].query == query
    assert search.history[0].results == results

    query = "What is langchain?"
    results = search(query=query)
    assert len(results) == 1
    assert 'title' in results[0]
    assert 'href' in results[0]
    assert 'body' in results[0]
    assert len(search.history) == 2
    assert search.history[1].query == query
    assert search.history[1].results == results


def test_DuckDuckGoSearch_caching():  # noqa
    """
    Test that searching DuckDuckGo based on same query returns same results with different uuid and
    timestamp.
    """
    query = "This is my fake query?"
    fake_results = [{'title': "fake results"}]
    search = DuckDuckGoSearch(top_n=1)
    # modify _history to mock a previous search based on a particular query
    search._history.append(SearchRecord(query=query, results=fake_results))
    response = search(query)
    assert response == fake_results
    assert len(search.history) == 2
    assert search.history[0].query == search.history[1].query
    assert search.history[0].results == search.history[1].results
    assert search.history[0].uuid != search.history[1].uuid
