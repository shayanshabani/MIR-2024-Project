import re
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize

class Preprocessor:

    def __init__(self, documents: list):
        """
        Initialize the class.

        Parameters
        ----------
        documents : list
            The list of documents to be preprocessed, path to stop words, or other parameters.
        """
        # TODO
        self.documents = documents
        self.stopwords = ['this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()

    def preprocess(self):
        """
        Preprocess the text using the methods in the class.

        Returns
        ----------
        List[str]
            The preprocessed documents.
        """
         # TODO
        preprocessed_documents = []
        for document in self.documents:
            for field in ['stars', 'genres', 'summaries']:
                if document[field] is not None:
                    for i in range(len(document[field])):
                        document[field][i] = self.remove_links(document[field][i])
                        document[field][i] = self.remove_punctuations(document[field][i])
                        document[field][i] = self.normalize(document[field][i])
                        document[field][i] = self.remove_stopwords(document[field][i])
                else:
                    document[field] = []
            preprocessed_documents.append(document)
        return preprocessed_documents

    def normalize(self, text: str):
        """
        Normalize the text by converting it to a lower case, stemming, lemmatization, etc.

        Parameters
        ----------
        text : str
            The text to be normalized.

        Returns
        ----------
        str
            The normalized text.
        """
        # TODO
        text = text.lower()
        tokens = self.tokenize(text)
        stemmed_tokens = [self.stemmer.stem(token) for token in tokens]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token) for token in stemmed_tokens]
        text = ' '.join(lemmatized_tokens)
        return text

    def remove_links(self, text: str):
        """
        Remove links from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with links removed.
        """
        # TODO
        patterns = ['https', 'http', 'www', '.ir', '.com', '.org', '@']
        for pattern in patterns:
            text = text.replace(pattern, ' ')
        return text

    def remove_punctuations(self, text: str):
        """
        Remove punctuations from the text.

        Parameters
        ----------
        text : str
            The text to be processed.

        Returns
        ----------
        str
            The text with punctuations removed.
        """
        # TODO
        text = re.sub(r'[^\w\s]', ' ', text)
        return text

    def tokenize(self, text: str):
        """
        Tokenize the words in the text.

        Parameters
        ----------
        text : str
            The text to be tokenized.

        Returns
        ----------
        list
            The list of words.
        """
        # TODO
        words = text.split()
        return words

    def remove_stopwords(self, text: str):
        """
        Remove stopwords from the text.

        Parameters
        ----------
        text : str
            The text to remove stopwords from.

        Returns
        ----------
        list
            The list of words with stopwords removed.
        """
        # TODO
        words = self.tokenize(text)
        filtered_words = [word for word in words if word not in self.stopwords]

        return ' '.join(filtered_words)

    def load_stopwords(self):
        """
        Load stopwords.

        Returns
        ----------
        set
            A set containing the stopwords.
        """
        with open('stopwords.txt', 'r') as f:
            stopwords = set(f.read().split())
        return stopwords

    def preprocess_query(self, documents):
        preprocessed_documents = []
        for document in documents:
            document = self.normalize(document)
            document = self.remove_links(document)
            document = self.remove_punctuations(document)
            document = self.remove_stopwords(document)
            preprocessed_documents.append(document)
        return preprocessed_documents