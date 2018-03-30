import string
import numpy as np

from .models import TermCountModel


class LSI:
    """Latent Semantic Indexing.
    """

    def __init__(self, docs, query, model=TermCountModel, rank_approximation=2, stopwords=None,
                 ignore_chars=string.punctuation):
        if stopwords is None:
            stopwords = []
        self.stopwords = stopwords
        self.ignore_chars = ignore_chars
        self.docs = list(map(self._parse, docs))
        self.words = self._get_words()
        self.query = self._parse_query(query)
        self.model = model
        self.rank_approximation = rank_approximation
        self.term_doc_matrix = self._build_term_doc_matrix()

    def _parse(self, text):
        translator = str.maketrans(self.ignore_chars, ' ' * len(self.ignore_chars))
        return list(map(str.lower,
                        filter(lambda w: w not in self.stopwords,
                               text.translate(translator).split())))

    def _parse_query(self, query):
        result = np.zeros(len(self.words))

        i = 0
        for word in sorted(self._parse(query)):
            while word > self.words[i]:
                i += 1
            if word == self.words[i]:
                result[i] += 1

        return result

    def _get_words(self):
        words = set()

        for doc in self.docs:
            words = words | set(doc)

        return sorted(words)

    def _build_term_doc_matrix(self):
        model = self.model(self.words, self.docs)
        return model.build()

    def _svd_with_dimensionality_reduction(self):
        u, s, v = np.linalg.svd(self.term_doc_matrix)
        s = np.diag(s)
        k = self.rank_approximation
        return u[:, :k], s[:k, :k], v[:, :k]

    def process(self):
        u_k, s_k, v_k = self._svd_with_dimensionality_reduction()

        q = self.query.T @ u_k @ np.linalg.pinv(s_k)
        d = self.term_doc_matrix.T @ u_k @ np.linalg.pinv(s_k)

        res = np.apply_along_axis(lambda row: self._sim(q, row), axis=1, arr=d)
        ranking = np.argsort(-res) + 1
        return ranking

    @staticmethod
    def _sim(x, y):
        return (x @ y) / (np.linalg.norm(x) * np.linalg.norm(y))
