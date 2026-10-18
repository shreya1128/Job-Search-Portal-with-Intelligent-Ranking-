"""Porter stemmer (NLTK implementation = supporting library) with a cache; the IR algorithms themselves are ours."""
from __future__ import annotations

from functools import lru_cache

from nltk.stem.porter import PorterStemmer

_ps = PorterStemmer()


@lru_cache(maxsize=200_000)
def stem(token: str) -> str:
    if len(token) <= 3 or not token.isalpha():
        return token
    return _ps.stem(token)
