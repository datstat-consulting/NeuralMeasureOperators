"""Text as positional feature-bearing empirical measures."""

from collections import Counter
import re

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .operators import FiniteMeasure


class TokenMeasureEncoder(TransformerMixin, BaseEstimator):
    """Fit a deterministic vocabulary and encode token sequences as measures."""

    def __init__(self, *, tokenizer=None, max_tokens=1024, lowercase=True):
        if not isinstance(max_tokens, int) or max_tokens < 1:
            raise ValueError("max_tokens must be a positive integer.")
        self.tokenizer = tokenizer
        self.max_tokens = max_tokens
        self.lowercase = lowercase

    def _tokens(self, document):
        if isinstance(document, str):
            document = document.lower() if self.lowercase else document
            tokenize = self.tokenizer or (lambda s: re.findall(r"\w+|[^\w\s]", s))
            tokens = list(tokenize(document))
        else:
            tokens = list(document)
            if self.lowercase:
                tokens = [token.lower() for token in tokens]
        if not tokens:
            raise ValueError("A token measure requires at least one token.")
        return tokens

    def fit(self, documents):
        frequencies = Counter(
            token for document in documents for token in self._tokens(document)
        )
        if not frequencies:
            raise ValueError("Provide documents to fit a vocabulary.")
        ranked = sorted(frequencies, key=lambda token: (-frequencies[token], token))
        self.vocabulary_ = {token: index for index, token in
                            enumerate(ranked[:self.max_tokens - 1], start=1)}
        return self

    def encode(self, document, *, embeddings=None, weights=None):
        """Use normalized token positions and one-hot or supplied embedding features."""
        check_is_fitted(self, "vocabulary_")
        ids = np.asarray([self.vocabulary_.get(token, 0) for token in self._tokens(document)])
        size = len(self.vocabulary_) + 1
        if embeddings is None:
            features = np.zeros((len(ids), size))
            features[np.arange(len(ids)), ids] = 1
        else:
            embeddings = np.asarray(embeddings, dtype=float)
            if embeddings.ndim != 2 or embeddings.shape[0] != size:
                raise ValueError("embeddings needs one row per vocabulary index, including unknown.")
            features = embeddings[ids]
        sites = np.arange(len(ids), dtype=float) / max(len(ids) - 1, 1)
        return FiniteMeasure(sites, features, weights)

    def transform(self, documents):
        return [self.encode(document) for document in documents]
