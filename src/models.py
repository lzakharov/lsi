from abc import ABC, abstractmethod
from typing import List

import numpy as np


class AbstractModel(ABC):
    def __init__(self, words: List[str], docs: List[List[str]]) -> None:
        self.words = words
        self.docs = docs

    @abstractmethod
    def build(self) -> np.ndarray:
        pass


class TermCountModel(AbstractModel):
    def build(self):
        model = np.zeros((len(self.words), len(self.docs)), dtype=int)

        for i, word in enumerate(self.words):
            for j, doc in enumerate(self.docs):
                model[i, j] = doc.count(word)

        return model


class TFIDFModel(TermCountModel):
    def build(self):
        term_count_model = super().build()
        model = np.zeros((len(self.words), len(self.docs)), dtype=float)

        for i, word in enumerate(self.words):
            for j, doc in enumerate(self.docs):
                tf = term_count_model[i, j] / len(doc)
                idf = np.log(sum(term_count_model[i] > 0))
                model[i, j] = tf * idf

        return model
