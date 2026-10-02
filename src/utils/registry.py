from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Generic, Iterable, Optional, TypeVar


T = TypeVar("T")


@dataclass(frozen=True)
class RegistryEntry(Generic[T]):
    name: str
    value: T


class Registry(Generic[T]):
    """A tiny name->object registry.

    Designed to keep the surface minimal while allowing future expansion to
    other components beyond models (e.g., data modules, trainers).
    """

    def __init__(self, registry_name: str):
        self._registry_name = registry_name
        self._items: Dict[str, T] = {}

    @property
    def name(self) -> str:
        return self._registry_name

    def register(self, item_name: str) -> Callable[[T], T]:
        if not item_name or not isinstance(item_name, str):
            raise ValueError(f"{self._registry_name}.register requires a non-empty string name")

        def _decorator(obj: T) -> T:
            if item_name in self._items:
                raise KeyError(
                    f"Duplicate registration in {self._registry_name}: '{item_name}'. "
                    f"Existing={self._items[item_name]} New={obj}"
                )
            self._items[item_name] = obj
            return obj

        return _decorator

    def get(self, item_name: str) -> Optional[T]:
        return self._items.get(item_name)

    def require(self, item_name: str) -> T:
        obj = self.get(item_name)
        if obj is None:
            available = ", ".join(sorted(self._items.keys()))
            raise KeyError(
                f"Unknown {self._registry_name} '{item_name}'. Available: [{available}]"
            )
        return obj

    def keys(self) -> Iterable[str]:
        return self._items.keys()

    def items(self) -> Iterable[RegistryEntry[T]]:
        for k, v in self._items.items():
            yield RegistryEntry(name=k, value=v)

