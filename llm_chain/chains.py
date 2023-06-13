"""TODO."""
from collections.abc import Callable
from typing import Any
import inspect

from llm_chain.base import Record, UsageRecord


class Chain:
    """TODO."""

    def __init__(self, links: list[Callable[[Any], Any]]):
        self._links = links

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """TODO."""
        if not self._links:
            return None
        result = self._links[0](*args, **kwargs)
        if len(self._links) > 1:
            for link in self._links[1:]:
                result = link(result)
        return result

    def __getitem__(self, index: int) -> Callable:
        return self._links[index]

    def __len__(self) -> int:
        return len(self._links)

    @property
    def history(self) -> list[Record]:
        """TODO."""
        histories = [link.history for link in self._links if _has_history(link)]
        # Edge-case: if the same model is used multiple times in the same chain (e.g. embedding
        # model to embed documents and then embed query to search documents) then we can't loop
        # through the chains because we'd be double-counting the history from those objects.
        # we have to build up a history and include the objects if they aren't already
        # to do this we'll use the uuid, and then sort by timestamp
        unique_records = []
        unique_uuids = set()
        for history in histories:
            for record in history:
                if record.uuid not in unique_uuids:
                    unique_records.append(record)
                    unique_uuids |= {record.uuid}
        return sorted(unique_records, key=lambda x: x.timestamp)

    @property
    def usage_history(self) -> list[UsageRecord]:
        """TODO."""
        return [x for x in self.history if isinstance(x, UsageRecord)]

    @property
    def total_tokens(self) -> str:
        """
        Returns the total number of tokens used by the all models during the chain/object's
        lifetime.

        Returns `None` if none of the models knows how to count tokens.
        """
        records = self.usage_history
        totals = [x.total_tokens for x in records if x.total_tokens]
        if not totals:
            return None
        return sum(totals)

    @property
    def total_cost(self) -> str:
        """
        Returns the total number of cost used by the all models during the chain/object's
        lifetime.

        Returns `None` if none of the models knows how to count cost.
        """
        records = self.usage_history
        totals = [x.cost for x in records if x.cost]
        if not totals:
            return None
        return sum(totals)


def _has_history(obj: object) -> bool:
    """TODO."""
    return _has_property(obj, property_name='history') and \
        isinstance(obj.history, list) and \
        len(obj.history) > 0 and \
        isinstance(obj.history[0], Record)


def _has_property(obj: object, property_name: str) -> bool:
    if inspect.isfunction(obj):
        return False
    return hasattr(obj, property_name)

    # from typing import get_type_hints
    # import inspect
    # for link in chain:
    #     if inspect.isfunction(link):
    #         type_hints = get_type_hints(link)
    #     else:
    #         type_hints = get_type_hints(link.__call__)
    #     return_type = type_hints.pop('return')
    #     print(f"{list(type_hints.values())[0]} -> {return_type}")
