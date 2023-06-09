"""Tests the utilities.py file."""
from time import sleep
import re
import pytest
from llm_chain.utilities import Timer

def test_timer_seconds():  # noqa
    with Timer() as timer:
        sleep(1.1)

    assert timer.interval
    assert re.match(pattern=r'1\.\d+ seconds', string=timer.formatted())
    assert str(timer) == timer.formatted()

    with pytest.raises(ValueError):  # noqa
        timer.formatted(units='days')
