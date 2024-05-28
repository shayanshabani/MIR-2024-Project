import numpy as np
from tqdm import tqdm

from Logic.core.word_embedding.fasttext_model import FastText
from sklearn.metrics import classification_report

class BasicClassifier:
    def __init__(self):
        # self.model = model
        # self.fasttext = FastText()
        pass

    def fit(self, x, y):
        # embeddings = [self.fasttext.get_sentence_vector(sentence) for sentence in tqdm(x, desc="Generating embeddings")]
        # self.model.fit(embeddings, y)
        pass

    def predict(self, x):
        # embeddings = [self.fasttext.get_sentence_vector(sentence) for sentence in tqdm(x, desc="Generating embeddings")]
        # return self.model.predict(embeddings)
        pass

    def prediction_report(self, x, y):
        # predictions = self.predict(x)
        # print(classification_report(y, predictions))
        pass

    def get_percent_of_positive_reviews(self, sentences):
        """
        Get the percentage of positive reviews in the given sentences
        Parameters
        ----------
        sentences: list
            The list of sentences to get the percentage of positive reviews
        Returns
        -------
        float
            The percentage of positive reviews
        """
        # predictions = self.predict(sentences)
        # positive_count = np.sum(predictions)
        # return (positive_count / len(predictions)) * 100
        pass

