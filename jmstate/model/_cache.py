from collections import OrderedDict
from typing import Any, Callable


class Cache:
    def __init__(
        self, cache_limit: int | None, keys: tuple[str, ...], *args: Any, **kwargs: Any
    ):
        """Inits the cache.

        Args:
            cache_limit (int | None): The cache limit, if None, means infinite.
            keys (tuple[str]): The keys of the cache.

        Raises:
            ValueError: If cache_limit is not None nor positive.
        """
        self.cache_limit = cache_limit
        self.keys = keys
        self.cache: dict[str, OrderedDict[tuple[int, ...], Any]] = {
            key: OrderedDict() for key in keys
        }

        if self.cache_limit is not None and self.cache_limit < 0:
            raise ValueError(
                f"cache_limit must be None or positive integer, got {self.cache_limit}"
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

    def clear_cache(self):
        """Clears the cached tensors."""
        self.cache = {key: OrderedDict() for key in self.keys}
