from collections import OrderedDict, defaultdict
from collections.abc import Callable
from typing import Any


class Cache:
    """Cache data."""

    cache_limit: int | None
    cache: defaultdict[str, OrderedDict[tuple[int, ...], Any]]  # type: ignore

    def __init__(self, cache_limit: int | None):
        """Initializes the cache.

        Args:
            cache_limit (int | None): The cache limit, if None, means infinite.

        Raises:
            ValueError: If cache_limit is not None nor positive.
        """
        self.cache_limit = cache_limit
        self.cache: defaultdict[str, OrderedDict[tuple[int, ...], Any]] = defaultdict(
            OrderedDict
        )

    def get_cache(self, name: str, key: Any, missing: Callable[[], Any]) -> Any:
        """Gets the cache [name][key].

        Args:
            name (str): The name of the cache info.
            key (Any): The key inside the current info.
            missing (Callable[[], Any]): The fallback function.

        Returns:
            Any: The cached element.
        """
        cache = self.cache[name]
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = missing()

        if self.cache_limit is not None and len(cache) > self.cache_limit:
            cache.popitem(last=False)

        return cache[key]

    def clear(self):
        """Clears the cache."""
        self.cache = defaultdict(OrderedDict)
