from __future__ import annotations
from abc import ABC, abstractmethod
from .types import SimpleGraphData

class GraphTokenizer(ABC):
    """Shared API for graph tokenization methods."""
    @abstractmethod
    def tokenize(self, data: SimpleGraphData):
        raise NotImplementedError

    @abstractmethod
    def decode(self, tokens):
        raise NotImplementedError

class TokenizerFactory:
    """Factory to fetch tokenizers without importing specific classes manually."""
    _tokenizers = {}

    @classmethod
    def register(cls, name):
        def inner(subclass):
            cls._tokenizers[name.lower()] = subclass
            return subclass
        return inner

    @classmethod
    def get_tokenizer(cls, name, **kwargs):
        if name.lower() not in cls._tokenizers:
            raise ValueError(f"Tokenizer {name} not found. Available: {list(cls._tokenizers.keys())}")
        return cls._tokenizers[name.lower()](**kwargs)