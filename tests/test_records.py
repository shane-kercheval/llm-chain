"""Tests Record objects in llm_chain.base.py."""
import re
from llm_chain.base import MessageRecord, EmbeddingsRecord, Record


def test_Record():  # noqa
    record = Record()
    assert isinstance(record.uuid, str)
    assert len(record.uuid) > 0
    assert isinstance(record.timestamp, str)
    assert re.match(pattern=r'\d\d\d\d-\d\d-\d\d \d\d:\d\d:\d\d', string=record.timestamp)

def test_EmbeddingsMetaData():  # noqa
    record_1 = EmbeddingsRecord(metadata={'model': 'test1'}, total_tokens=10)
    assert isinstance(record_1.uuid, str)
    assert len(record_1.uuid) > 0
    assert isinstance(record_1.timestamp, str)
    assert record_1.metadata['model'] == 'test1'
    assert record_1.total_tokens == 10
    assert record_1.cost is None

    record_2 = EmbeddingsRecord(metadata={'model': 'test2'}, total_tokens=5, cost=1)
    assert isinstance(record_2.uuid, str)
    assert len(record_2.uuid) > 0
    assert isinstance(record_2.timestamp, str)
    assert record_2.metadata['model'] == 'test2'
    assert record_2.total_tokens == 5
    assert record_2.cost == 1
    assert record_1.uuid != record_2.uuid

def test_MessageMetaData():  # noqa
    record_1 = MessageRecord(
        prompt='prompt',
        response='response',
        metadata={'model': 'test1'},
        total_tokens=10,
    )
    assert isinstance(record_1.uuid, str)
    assert len(record_1.uuid) > 0
    assert isinstance(record_1.timestamp, str)
    assert record_1.metadata['model'] == 'test1'
    assert record_1.total_tokens == 10
    assert record_1.cost is None

    record_2 = MessageRecord(
        prompt='prompt',
        response='response',
        metadata={'model': 'test2'},
        total_tokens=5,
        cost=1,
    )
    assert isinstance(record_2.uuid, str)
    assert len(record_2.uuid) > 0
    assert isinstance(record_2.timestamp, str)
    assert record_2.metadata['model'] == 'test2'
    assert record_2.total_tokens == 5
    assert record_2.cost == 1
    assert record_1.uuid != record_2.uuid
