from collections.abc import MutableMapping
from typing import TypeVar, Generic, Callable, Dict, Any, Optional, Union

K = TypeVar('K')
V = TypeVar('V')
DEFAULT = object()

class BiDict(MutableMapping, Generic[K, V]):
    def __init__(
        self,
        mirror: Optional["BiDict[V, K]"] = None,
        val_btype: type = set,
        key_map: Optional[Callable[[K], Any]] = None
    ) -> None:
        self._dict: Dict[Any, Union[set, list]] = {}
        self.mirror: Optional["BiDict[V, K]"] = mirror
        self.val_btype = val_btype
        self.key_map = key_map if key_map is not None else lambda x: x

        if mirror is not None:
            mirror.mirror = self

    def __contains__(self, key: K) -> bool:
        return self.key_map(key) in self._dict

    def __getitem__(self, key: K) -> Union[set, list]:
        mapped_key = self.key_map(key)
        return self._dict.get(mapped_key, self.val_btype())

    def __setitem__(self, key: K, value: V, inv: bool = False) -> None:
        mapped_key = self.key_map(key)
        if self.val_btype is set:
            self._dict.setdefault(mapped_key, set()).add(value)
        elif self.val_btype is list:
            self._dict.setdefault(mapped_key, list()).append(value)
        else:
            raise TypeError(f"Unsupported val_btype: {self.val_btype}")

        if not inv and self.mirror:
            self.mirror.__setitem__(value, key, inv=True)

    def __delitem__(self, key: K) -> None:
        mapped_key = self.key_map(key)
        if mapped_key not in self._dict:
            raise KeyError(key)

        vals = set(self._dict[mapped_key])
        for val in vals:
            if self.mirror and key in self.mirror[val]:
                self.mirror[val].remove(key)
                if not self.mirror[val]:
                    del self.mirror[val]

        del self._dict[mapped_key]

    def __iter__(self):
        return iter(self._dict)

    def __len__(self) -> int:
        return len(self._dict)

    def __repr__(self) -> str:
        return repr(self._dict)

    def pop(self, key: K, default: Any = DEFAULT) -> Union[set, list, Any]:
        if key in self or default is DEFAULT:
            mapped_key = self.key_map(key)
            value = self[key]
            del self[key]
            return value
        return default
