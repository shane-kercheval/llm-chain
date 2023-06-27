"""Test Session class."""
from time import sleep
import pytest
from llm_chain.base import Chain, HistoricalUsageRecords, MessageRecord, Record, Session, \
    UsageRecord


class MockHistoricalUsageRecords(HistoricalUsageRecords):
    """TODO."""

    def __init__(self, mock_id: str) -> None:
        super().__init__()
        # mock_id is used to tset that the correct chain is called
        self.mock_id = mock_id
        self.records = []

    def __call__(self, record: Record) -> Record:  # noqa
        self.records.append(record)
        return record, self.mock_id

    @property
    def history(self) -> list[UsageRecord]:  # noqa
        return self.records

def test_Session():  # noqa
    session = Session()
    with pytest.raises(ValueError):  # noqa: PT011
        session('test')
    assert session.history == []
    assert session.usage_history == []
    assert session.message_history == []
    assert session.cost is None
    assert session.total_tokens is None
    assert len(session) == 0

    session.append(chain=Chain(links=[]))
    assert session('test') is None
    assert session.history == []
    assert session.usage_history == []
    assert session.message_history == []
    assert session.cost is None
    assert session.total_tokens is None
    assert len(session) == 1

    # test chain with a link that doesn't have a history property
    session.append(chain=Chain(links=[lambda x: x]))
    assert session('test') == 'test'
    assert session.history == []
    assert session.usage_history == []
    assert session.message_history == []
    assert session.cost is None
    assert session.total_tokens is None
    assert len(session) == 2

    record_a = UsageRecord(metadata={'id': 'record_a'}, total_tokens=None, cost=None)
    sleep(0.001)
    record_b = UsageRecord(metadata={'id': 'record_b'}, total_tokens=100, cost=0.01)
    sleep(0.001)
    record_c = Record(metadata={'id': 'record_d'})
    sleep(0.001)
    record_d = Record(metadata={'id': 'record_e'})
    sleep(0.001)
    record_e = MessageRecord(
        metadata={'id': 'record_e'},
        prompt='prompt',
        response='response',
        cost=0.5,
        total_tokens=103,
    )

    session.append(chain=Chain(links=[MockHistoricalUsageRecords(mock_id='mock_a')]))
    return_value, mock_id = session(record_a)
    assert return_value == record_a
    assert mock_id == 'mock_a'
    assert session.history == [record_a]
    assert session.usage_history == [record_a]
    assert session.message_history == []
    assert session.cost is None
    assert session.total_tokens is None
    assert len(session) == 3

    # if we add the same record it should be ignored
    return_value, mock_id = session(record_a)
    assert return_value == record_a
    assert mock_id == 'mock_a'
    assert session.history == [record_a]
    assert session.usage_history == [record_a]
    assert session.message_history == []
    assert session.cost is None
    assert session.total_tokens is None
    assert len(session) == 3

    return_value, mock_id = session(record_b)
    assert return_value == record_b
    assert mock_id == 'mock_a'
    assert session.history == [record_a, record_b]
    assert session.usage_history == [record_a, record_b]
    assert session.message_history == []
    assert session.cost == 0.01
    assert session.total_tokens == 100
    assert len(session) == 3

    # add record `e` out of order; later, ensure the correct order is returned
    session.append(chain=Chain(links=[MockHistoricalUsageRecords(mock_id='mock_b')]))
    return_value, mock_id = session(record_e)
    assert return_value == record_e
    assert mock_id == 'mock_b'
    assert session.history == [record_a, record_b, record_e]
    assert session.usage_history == [record_a, record_b, record_e]
    assert session.message_history == [record_e]
    assert session.cost == 0.51
    assert session.total_tokens == 203
    assert len(session) == 4

    # adding the same record to a new link should not double-count
    return_value, mock_id = session(record_b)
    assert return_value == record_b
    assert mock_id == 'mock_b'
    assert session.history == [record_a, record_b, record_e]
    assert session.usage_history == [record_a, record_b, record_e]
    assert session.message_history == [record_e]
    assert session.cost == 0.51
    assert session.total_tokens == 203
    assert len(session) == 4

    # add record `d` out of order; later, ensure the correct order is returned
    return_value, mock_id = session(record_d)
    assert return_value == record_d
    assert mock_id == 'mock_b'
    assert session.history == [record_a, record_b, record_d, record_e]
    assert session.usage_history == [record_a, record_b, record_e]
    assert session.message_history == [record_e]
    assert session.cost == 0.51
    assert session.total_tokens == 203
    assert len(session) == 4

    # add record `c` out of order; c should be returned before d
    return_value, mock_id = session(record_c)
    assert return_value == record_c
    assert mock_id == 'mock_b'
    assert session.history == [record_a, record_b, record_c, record_d, record_e]
    assert session.usage_history == [record_a, record_b, record_e]
    assert session.message_history == [record_e]
    assert session.cost == 0.51
    assert session.total_tokens == 203
    assert len(session) == 4
