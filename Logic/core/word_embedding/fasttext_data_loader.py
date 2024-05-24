import json

import nltk
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import re

from nltk.corpus import stopwords
nltk.download('stopwords')
from sklearn.preprocessing import LabelEncoder


def preprocess_text(text, minimum_length=1, stopword_removal=True, stopwords_domain=[], lower_case=True,
                       punctuation_removal=True):
    """
    preprocess text by removing stopwords, punctuations, and converting to lowercase, and also filter based on a min length
    for stopwords use nltk.corpus.stopwords.words('english')
    for punctuations use string.punctuation

    Parameters
    ----------
    text: str
        text to be preprocessed
    minimum_length: int
        minimum length of the token
    stopword_removal: bool
        whether to remove stopwords
    stopwords_domain: list
        list of stopwords to be removed base on domain
    lower_case: bool
        whether to convert to lowercase
    punctuation_removal: bool
        whether to remove punctuations
    """
    if punctuation_removal:
        text = re.sub(r'[^\w\s]', ' ', text)

    words = text.split()

    minimum_tokens = [word for word in words if len(word) >= minimum_length]
    text = ' '.join(minimum_tokens)

    if stopword_removal:
        stop_words = set(stopwords.words('english') + stopwords_domain)
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)

    if lower_case:
        text = text.lower()

    return text

class FastTextDataLoader:
    """
    This class is designed to load and pre-process data for training a FastText model.

    It takes the file path to a data source containing movie information (synopses, summaries, reviews, titles, genres) as input.
    The class provides methods to read the data into a pandas DataFrame, pre-process the text data, and create training data (features and labels)
    """
    def __init__(self, file_path):
        """
        Initializes the FastTextDataLoader class with the file path to the data source.

        Parameters
        ----------
        file_path: str
            The path to the file containing movie information.
        """
        self.file_path = file_path

    def read_data_to_df(self):
        """
        Reads data from the specified file path and creates a pandas DataFrame containing movie information.

        You can use an IndexReader class to access the data based on document IDs.
        It extracts synopses, summaries, reviews, titles, and genres for each movie.
        The extracted data is then stored in a pandas DataFrame with appropriate column names.

        Returns
        ----------
            pd.DataFrame: A pandas DataFrame containing movie information (synopses, summaries, reviews, titles, genres).
        """
        with open('../../IMDB_crawled.json', 'r') as f:
            json_data = f.read()

        crawled_movies = json.loads(json_data)

        data = []

        for movie in crawled_movies:
            if movie['genres'] is not None and movie['reviews'] is not None and movie['synposis'] is not None and movie[
                'summaries'] is not None:

                each = {'titles': movie['title'], 'synopsis': movie['synposis'][0],
                        'summaries': ' '.join(movie['summaries'])}

                full_review = ''

                for review in movie['reviews']:
                    full_review += review[0]
                    full_review += ' '
                each['reviews'] = full_review

                each['genres'] = ' '.join(movie['genres'])

                data.append(each)

        df = pd.DataFrame(data, columns=['synopsis', 'summaries', 'reviews', 'titles', 'genres'])
        return df


    def create_train_data(self):
        """
        Reads data using the read_data_to_df function, pre-processes the text data, and creates training data (features and labels).

        Returns:
            tuple: A tuple containing two NumPy arrays: X (preprocessed text data) and y (encoded genre labels).
        """
        df = self.read_data_to_df()
        another = []
        for _, data in df.iterrows():
            which = data['synopsis'] + ' ' + data['summaries'] + ' ' + data['reviews'] + ' ' + data['titles']
            which = preprocess_text(which)
            x = {
                'X': which,
                'y': data['genres']
            }
            another.append(x)

        final_df = pd.DataFrame(another, columns=['X', 'y'])

        le = LabelEncoder()
        final_df['genre_encoded'] = le.fit_transform(final_df['y'])
        X = final_df['X'].values
        y = final_df['genre_encoded'].values
        return X, y

