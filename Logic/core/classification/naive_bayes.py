import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

from Logic.core.classification.basic_classifier import BasicClassifier
from Logic.core.classification.data_loader import ReviewLoader


class NaiveBayes(BasicClassifier):
    def __init__(self, count_vectorizer, alpha=1):
        super().__init__()
        self.cv = count_vectorizer
        self.num_classes = None
        self.classes = None
        self.number_of_features = None
        self.number_of_samples = None
        self.prior = None
        self.feature_probabilities = None
        self.log_probs = None
        self.alpha = alpha

    def fit(self, x, y):
        """
        Fit the features and the labels
        Calculate prior and feature probabilities

        Parameters
        ----------
        x: np.ndarray
            An m * n matrix - m is count of docs and n is embedding size

        y: np.ndarray
            The real class label for each doc

        Returns
        -------
        self
            Returns self as a classifier
        """
        labels = np.unique(y)
        num_labels = len(np.unique(y))
        num_samples, num_features = x.shape
        priors = np.zeros(num_labels)
        feature_probs = np.zeros((num_labels, num_features))

        for idx in range(num_labels):
            label = labels[idx]
            label_samples = x[y == label]
            label_sample_count = label_samples.shape[0]
            priors[idx] = label_sample_count / num_samples
            feature_sum = np.sum(label_samples, axis=0)
            feature_probs[idx] = (feature_sum + self.alpha) / (np.sum(feature_sum) + self.alpha)

        log_feature_probs = np.log(feature_probs)

        self.num_classes = num_labels
        self.classes = labels
        self.number_of_samples = num_samples
        self.number_of_features = num_features
        self.prior = priors
        self.feature_probabilities = feature_probs
        self.log_probs = log_feature_probs

        return self

    def predict(self, x):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        Returns
        -------
        np.ndarray
            Return the predicted class for each doc
            with the highest probability (argmax)
        """
        scores = np.zeros((x.shape[0], self.num_classes))

        for i in range(self.num_classes):
            scores[:, i] = np.dot(x, self.log_probs[i, :]) + np.log(self.prior[i])

        predicted_classes = np.argmax(scores, axis=1)
        return predicted_classes

    def prediction_report(self, x, y):
        """
        Parameters
        ----------
        x: np.ndarray
            An k * n matrix - k is count of docs and n is embedding size
        y: np.ndarray
            The real class label for each doc
        Returns
        -------
        str
            Return the classification report
        """
        y_pred = self.predict(x)
        report = classification_report(y, y_pred)
        return report

    def get_percent_of_positive_reviews(self, sentences):
        """
        You have to override this method because we are using a different embedding method in this class.
        """
        df = pd.DataFrame(sentences, columns=['review'])
        review_tokens = df['review'].to_numpy()
        x = self.cv.fit_transform(review_tokens)
        sentiments = self.predict(x)
        count = 0
        for sentiment in sentiments:
            if sentiment == 1:
                count += 1
        percent = (count / len(sentiments)) * 100
        return percent

# F1 Accuracy : 85%
if __name__ == '__main__':
    """
    First, find the embeddings of the reviews using the CountVectorizer, then fit the model with the training data.
    Finally, predict the test data and print the classification report
    You can use scikit-learn's CountVectorizer to find the embeddings.
    """
    data_loader = ReviewLoader("data/IMDB_Dataset.csv")
    x, y = data_loader.load_data()
    x_train, x_test, y_train, y_test = data_loader.split_data(naive=True)
    count_vectorizer = CountVectorizer(max_features=25000)
    x_train = count_vectorizer.fit_transform(x_train)
    x_test = count_vectorizer.transform(x_test)
    naive_bayes = NaiveBayes(count_vectorizer)
    naive_bayes.fit(x_train.toarray(), y_train)
    print(naive_bayes.prediction_report(x_test.toarray(), y_test))