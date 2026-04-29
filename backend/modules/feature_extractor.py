"""
feature_extractor.py
--------------------
Builds and manages the TF-IDF vector space for internship matching.
Supports incremental updates and serialisation so the model can be
re-used without retraining on every request.
"""

import pickle
import os
import numpy as np
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Path where the serialised vectorizer is cached
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tfidf_cache.pkl")


class FeatureExtractor:
    """
    Wraps a TF-IDF vectorizer and the pre-computed internship matrix.

    Usage
    -----
    fe = FeatureExtractor()
    fe.fit(corpus)                          # train on internship descriptions
    query_vec = fe.transform_query(text)    # vectorise user profile
    scores = fe.similarity_scores(query_vec)
    """

    def __init__(self):
        self.vectorizer: TfidfVectorizer = TfidfVectorizer(
            ngram_range=(1, 2),      # unigrams + bigrams for richer matching
            max_features=8000,
            sublinear_tf=True,       # replace TF with 1+log(TF) to reduce bias
            min_df=1,
            analyzer="word",
            token_pattern=r"[a-zA-Z][a-zA-Z0-9\-\.\/]{1,}",  # keeps "node.js" etc.
        )
        self.corpus_matrix = None   # sparse matrix: (n_internships × n_features)
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training / fitting
    # ------------------------------------------------------------------

    def fit(self, corpus: List[str]) -> None:
        """
        Fit the TF-IDF vectorizer on the internship corpus and cache
        the resulting document matrix for fast similarity lookups.

        Parameters
        ----------
        corpus : list of preprocessed internship text strings
        """
        self.corpus_matrix = self.vectorizer.fit_transform(corpus)
        self.is_fitted = True

    def save(self, path: str = None) -> None:
        """Persist vectorizer + corpus matrix to disk."""
        path = path or _CACHE_PATH
        with open(path, "wb") as f:
            pickle.dump({
                "vectorizer": self.vectorizer,
                "corpus_matrix": self.corpus_matrix,
            }, f)

    def load(self, path: str = None) -> bool:
        """
        Load a previously serialised model from disk.
        Returns True on success, False if no cache exists.
        """
        path = path or _CACHE_PATH
        if not os.path.exists(path):
            return False
        with open(path, "rb") as f:
            state = pickle.load(f)
        self.vectorizer = state["vectorizer"]
        self.corpus_matrix = state["corpus_matrix"]
        self.is_fitted = True
        return True

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def transform_query(self, text: str) -> np.ndarray:
        """
        Convert a user profile string into a TF-IDF vector using the
        already-fitted vocabulary.

        Returns a (1 × n_features) sparse matrix.
        """
        if not self.is_fitted:
            raise RuntimeError("FeatureExtractor must be fitted before calling transform_query.")
        return self.vectorizer.transform([text])

    def similarity_scores(self, query_vector) -> np.ndarray:
        """
        Compute cosine similarity between the query vector and all
        internship vectors in the corpus matrix.

        Returns a 1-D numpy array of shape (n_internships,).
        """
        scores = cosine_similarity(query_vector, self.corpus_matrix)
        return scores.flatten()

    def get_top_feature_terms(self, query_vector, n: int = 10) -> List[str]:
        """
        Return the top-n terms that contributed most to the query vector.
        Useful for generating 'why this internship is recommended' explanations.
        """
        if not self.is_fitted:
            return []
        feature_names = np.array(self.vectorizer.get_feature_names_out())
        query_arr = query_vector.toarray().flatten()
        top_indices = query_arr.argsort()[::-1][:n]
        return [feature_names[i] for i in top_indices if query_arr[i] > 0]
