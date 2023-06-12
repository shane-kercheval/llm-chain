"""TODO."""
from collections.abc import Callable
from typing import Any
import inspect

from llm_chain.base import UsageRecord


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

    def _get_unique_records(self) -> list[UsageRecord]:
        histories = [link.history for link in self._links if _has_usage_history(link)]
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
    def total_tokens(self) -> str:
        """
        Returns the total number of tokens used by the all models during the chain/object's
        lifetime.

        Returns `None` if none of the models knows how to count tokens.
        """
        records = self._get_unique_records()
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
        records = self._get_unique_records()
        totals = [x.cost for x in records if x.cost]
        if not totals:
            return None
        return sum(totals)


    # @property
    # def total_cost(self) -> str:
    #     """
    #     Returns the total cost associated with usage across all models during the chain/object's
    #     lifetime.

    #     Returns `None` if none of the models know how to count costs.
    #     """
    #     links = [
    #         link for link in self.chain
    #         if _has_property(link, property_name='total_cost') and link.total_cost
    #     ]
    #     # Edge-case: if the same model is used multiple times in the same chain (e.g. embedding
    #     # model) we can't loop through the chains because we'd be double-counting the totals from
    #     # objects that are already included multiple times
    #     # we have to build up a list of objects and include the objects if they aren't already
    #     unique_links = []
    #     for link in links:
    #         if link not in unique_links:
    #             unique_links.append(link)

    #     if not unique_links:
    #         return None

    #     return sum(x.total_cost for x in unique_links)

def _has_usage_history(obj: object) -> bool:
    return _has_property(obj, property_name='history') and \
        isinstance(obj.history, list) and \
        len(obj.history) > 0 and \
        isinstance(obj.history[0], UsageRecord)


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
