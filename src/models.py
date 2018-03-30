import numpy as np


class TermCountModel:
    def __init__(self, words, docs):
        self.words = words
        self.docs = docs

    def build(self):
        model = np.zeros((len(self.words), len(self.docs)), dtype=int)

        for i, word in enumerate(self.words):
            for j, doc in enumerate(self.docs):
                model[i, j] = doc.count(word)

        return model


class TFIDFModel(TermCountModel):
    def build(self):
        model = super().build()

        for i, word in enumerate(self.words):
            for j, doc in enumerate(self.docs):
                model[i, j] = (model[i, j] / len(doc)) * np.log(sum(model[i] > 0))

        return model
