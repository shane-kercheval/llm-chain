"""TODO."""
from collections.abc import Callable
from typing import Any
import inspect


class Chain:
    """TODO."""

    def __init__(self, chain: list[Callable[[Any], Any]]):
        self.chain = chain

    def __call__(self, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        """TODO."""
        if not self.chain:
            return None
        result = self.chain[0](*args, **kwargs)
        if len(self.chain) > 1:
            for link in self.chain[1:]:
                result = link(result)
        return result

    @property
    def total_tokens(self) -> str:
        """
        Returns the total number of tokens used by the all models during the chain/object's
        lifetime.

        #TODO: VERIFY/UPDATREReturns `None` none of the models does not know how to count tokens.
        """
        links = [
            link for link in self.chain
            if _has_property(link, property_name='total_tokens') and link.total_tokens
        ]
        # edge case: the same model is used multiple times in the same chain (e.g. embedding model)
        # we can't loop through the chains because we'd be double-counting the totals from objects
        # that are included multiple times
        # we have to build up a list of objects and include the objects if they aren't already
        unique_links = []
        for link in links:
            if link not in unique_links:
                unique_links.append(link)

        if not unique_links:
            return None

        return sum(x.total_tokens for x in unique_links)

    @property
    def total_cost(self) -> str:
        """
        Returns the total cost associated with usage across all models during the chain/object's
        lifetime.

        #TODO: VERIFY/UPDATREReturns Returns `None` if the model does not know how to count costs.
        """
        links = [
            link for link in self.chain
            if _has_property(link, property_name='total_cost') and link.total_cost
        ]
        # edge case: the same model is used multiple times in the same chain (e.g. embedding model)
        # we can't loop through the chains because we'd be double-counting the totals from objects
        # that are included multiple times
        # we have to build up a list of objects and include the objects if they aren't already
        unique_links = []
        for link in links:
            if link not in unique_links:
                unique_links.append(link)

        if not unique_links:
            return None

        return sum(x.total_cost for x in unique_links)

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
