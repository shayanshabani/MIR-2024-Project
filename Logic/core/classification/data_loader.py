import numpy as np
import pandas as pd
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from Logic.core.word_embedding.fasttext_model import FastText


class ReviewLoader:
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.fasttext_model = None
        self.review_tokens = []
        self.sentiments = []
        self.embeddings = []

    def load_data(self):
        """
        Load the data from the csv file and preprocess the text. Then save the normalized tokens and the sentiment labels.
        Also, load the fasttext model.
        """
        self.fasttext_model = FastText()
        self.fasttext_model.prepare(None, mode="load", path='../word_embedding/data/FastText_model.bin')

        data_frame = pd.read_csv(self.file_path)

        label_encoder = LabelEncoder()
        data_frame['sentiment'] = label_encoder.fit_transform(data_frame['sentiment'])

        self.review_tokens = data_frame['review'].to_numpy()
        self.sentiments = data_frame['sentiment'].to_numpy()
        return self.review_tokens, self.sentiments

    def get_embeddings(self):
        """
        Get the embeddings for the reviews using the fasttext model.
        """
        embeddings_list = []
        for token in tqdm.tqdm(self.review_tokens):
            embedding = self.fasttext_model.get_query_embedding(token)
            embeddings_list.append(embedding)

        self.embeddings = np.array(embeddings_list)

    def split_data(self, test_data_ratio=0.2, naive=False):
        """
        Split the data into training and testing data.

        Parameters
        ----------
        test_data_ratio: float
            The ratio of the test data
        naive: bool
            The type of classification for splitting data
        Returns
        -------
        np.ndarray, np.ndarray, np.ndarray, np.ndarray
            Return the training and testing data for the embeddings and the sentiments.
            in the order of x_train, x_test, y_train, y_test
        """
        if naive is False:
            x_train, x_test, y_train, y_test = train_test_split(self.embeddings, self.sentiments, test_size=test_data_ratio, random_state=42)
            return x_train, x_test, y_train, y_test
        else:
            x_train, x_test, y_train, y_test = train_test_split(self.review_tokens, self.sentiments, test_size=test_data_ratio, random_state=42)
            return x_train, x_test, y_train, y_test
