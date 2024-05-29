import time
import os
import json
import copy
from indexes_enum import Indexes
from Logic.core.utility.preprocess import Preprocessor

class Index:
    def __init__(self, preprocessed_documents: list):
        """
        Create a class for indexing.
        """

        self.preprocessed_documents = preprocessed_documents

        self.index = {
            Indexes.DOCUMENTS.value: self.index_documents(),
            Indexes.STARS.value: self.index_stars(),
            Indexes.GENRES.value: self.index_genres(),
            Indexes.SUMMARIES.value: self.index_summaries(),
        }

    def index_documents(self):
        """
        Index the documents based on the document ID. In other words, create a dictionary
        where the key is the document ID and the value is the document.

        Returns
        ----------
        dict
            The index of the documents based on the document ID.
        """

        document_index = {}
        #         TODO
        for document in self.preprocessed_documents:
            document_index[document['id']] = document
        return document_index

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        star_index = {}
        for document in self.preprocessed_documents:
            term_freq = {}

            for star in document['stars']:
                star_tokens = star.split()

                for term in star_tokens:
                    if term not in term_freq:
                        term_freq[term] = 0
                    term_freq[term] += 1

            for term, freq in term_freq.items():
                if term not in star_index:
                    star_index[term] = {}
                star_index[term][document['id']] = freq
        return star_index

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        #         TODO
        genre_index = {}
        for document in self.preprocessed_documents:
            genre_freq = {}
            for genre in document['genres']:
                if genre not in genre_freq:
                    genre_freq[genre] = 0
                genre_freq[genre] += 1

            for genre, freq in genre_freq.items():
                if genre not in genre_index:
                    genre_index[genre] = {}
                genre_index[genre][document['id']] = freq
        return genre_index

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries. You should also store each terms' tf in each document.
            So the index type is: {term: {document_id: tf}}
        """

        summary_index = {}
        #         TODO
        for document in self.preprocessed_documents:
            if document['id'] == 'tt0316654':
                print('fuckkkkkkkkkkkkkkkkk')
            term_freq = {}
            for summary in document['summaries']:
                summary_tokens = summary.split()

                for term in summary_tokens:
                    if term not in term_freq:
                        term_freq[term] = 0
                    term_freq[term] += 1

            for term, freq in term_freq.items():
                if term not in summary_index:
                    summary_index[term] = {}
                summary_index[term][document['id']] = freq
        return summary_index

    def get_posting_list(self, word: str, index_type: str):
        """
        get posting_list of a word

        Parameters
        ----------
        word: str
            word we want to check
        index_type: str
            type of index we want to check (documents, stars, genres, summaries)

        Return
        ----------
        list
            posting list of the word (you should return the list of document IDs that contain the word and ignore the tf)
        """

        try:
            #         TODO
            if index_type == 'documents':
                return sorted([document['id'] for document in self.preprocessed_documents if word in document])
            elif index_type == 'stars':
                return sorted(list(self.index['stars'].get(word, {}).keys()))
            elif index_type == 'genres':
                return sorted(list(self.index['genres'].get(word, {}).keys()))
            elif index_type == 'summaries':
                return sorted(list(self.index['summaries'].get(word, {}).keys()))

        except Exception as e:
            print('Error: ', e)
            return []

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO
        # documents
        document_index = self.index['documents']
        document_index[document['id']] = document
        self.index['documents'] = document_index
        # stars
        star_index = self.index['stars']
        term_freq = {}
        for star in document['stars']:
            star_tokens = star.split()

            for term in star_tokens:
                if term not in term_freq:
                    term_freq[term] = 0
                term_freq[term] += 1

        for term, freq in term_freq.items():
            if term not in star_index:
                star_index[term] = {}
            star_index[term][document['id']] = freq
        self.index['stars'] = star_index
        # genres
        genre_index = self.index['genres']
        genre_freq = {}
        for genre in document['genres']:
            if genre not in genre_freq:
                genre_freq[genre] = 0
            genre_freq[genre] += 1

        for genre, freq in genre_freq.items():
            if genre not in genre_index:
                genre_index[genre] = {}
            genre_index[genre][document['id']] = freq
        self.index['genres'] = genre_index
        # summaries
        summary_index = self.index['summaries']
        term_freq = {}
        for summary in document['summaries']:
            summary_tokens = summary.split()

            for term in summary_tokens:
                if term not in term_freq:
                    term_freq[term] = 0
                term_freq[term] += 1

        for term, freq in term_freq.items():
            if term not in summary_index:
                summary_index[term] = {}
            summary_index[term][document['id']] = freq


    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        #         TODO
        # documents
        document_index = self.index['documents']
        if document_id in document_index:
            document_index.pop(document_id)
        # stars
        to_be_removed = []
        star_index = self.index['stars']
        for star in star_index:
            if document_id in star_index[star]:
                star_index[star].pop(document_id)
                if len(star_index[star]) == 0 and (star not in to_be_removed):
                    to_be_removed.append(star)
        for to_be in to_be_removed:
            star_index.pop(to_be)
        self.index['stars'] = star_index
        # genres
        to_be_removed = []
        genre_index = self.index['genres']
        for genre in genre_index:
            if document_id in genre_index[genre]:
                genre_index[genre].pop(document_id)
                if len(genre_index[genre]) == 0 and (genre not in to_be_removed):
                    to_be_removed.append(genre)
        for to_be in to_be_removed:
            genre_index.pop(to_be)
        self.index['genres'] = genre_index
        # summaries
        to_be_removed = []
        summary_index = self.index['summaries']
        for summary in summary_index:
            if document_id in summary_index[summary]:
                summary_index[summary].pop(document_id)
                if len(summary_index[summary]) == 0 and (summary not in to_be_removed):
                    to_be_removed.append(summary)
        for to_be in to_be_removed:
            summary_index.pop(to_be)
        self.index['summaries'] = summary_index


    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['emma', 'mark'],
            'genres': ['drama', 'crime'],
            'summaries': ['fighter']
        }
        # for star in dummy_document['stars']:
        #     if star not in self.index['stars']:
        #         self.index['stars'][star] = {}
        # for genre in dummy_document['genres']:
        #     if genre not in self.index['genres']:
        #         self.index['genres'][genre] = {}


        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['emma']).difference(set(index_before_add[Indexes.STARS.value]['emma']))
                != {dummy_document['id']}):
            print('Add is incorrect, emma')
            return

        if (set(index_after_add[Indexes.STARS.value]['mark']).difference(set(index_before_add[Indexes.STARS.value]['mark']))
                != {dummy_document['id']}):
            print('Add is incorrect, mark')
            return
        if (set(index_after_add[Indexes.GENRES.value]['drama']).difference(set(index_before_add[Indexes.GENRES.value]['drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['crime']).difference(set(index_before_add[Indexes.GENRES.value]['crime']))
                != {dummy_document['id']}):
            print('Add is incorrect, crime')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['fighter']).difference(set(index_before_add[Indexes.SUMMARIES.value]['fighter']))
                != {dummy_document['id']}):
            print('Add is incorrect, fighter')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')


    def store_index(self, path: str, index_name: str = None):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_name: str
            name of index we want to store (documents, stars, genres, summaries)
        """

        if index_name not in self.index:
            raise ValueError('Invalid index name')

        # TODO
        json_object = json.dumps(self.index[index_name], indent=4)
        with open(path, 'w') as f:
            f.write(json_object)


    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        #         TODO
        with open(path, 'r') as f:
            json_data = f.read()
        self.index[path[6:-11]] = json.loads(json_data)


    def check_if_index_loaded_correctly(self, index_type: str, loaded_index: dict):
        """
        Check if the index is loaded correctly

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        loaded_index : dict
            The loaded index

        Returns
        ----------
        bool
            True if index is loaded correctly, False otherwise
        """

        return self.index[index_type] == loaded_index

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'fighter'):
        """
        Checks if the indexing is good. Do not change this function. You can use this
        function to check if your indexing is correct.

        Parameters
        ----------
        index_type : str
            Type of index to check (documents, stars, genres, summaries)
        check_word : str
            The word to check in the index

        Returns
        ----------
        bool
            True if indexing is good, False otherwise
        """

        # brute force to check check_word in the summaries
        start = time.time()
        docs = []
        another = []
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field.split():
                    docs.append(document['id'])
                    another.append(document)
                    break

            # if we have found 3 documents with the word, we can break
            if len(docs) == 3:
                break

        end = time.time()
        brute_force_time = end - start

        # check by getting the posting list of the word
        start = time.time()
        # TODO: based on your implementation, you may need to change the following line
        posting_list = self.get_posting_list(check_word, index_type)

        end = time.time()
        implemented_time = end - start

        print('Brute force time: ', brute_force_time)
        print('Implemented time: ', implemented_time)

        if set(docs).issubset(set(posting_list)):
            print('Indexing is correct')

            if implemented_time <= brute_force_time:
                print('Indexing is good')
                return True
            else:
                print('Indexing is bad')
                return False
        else:
            print('Indexing is wrong')
            print(set(docs))
            return False

# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods

with open('../../IMDB_crawled.json', 'r') as f:
    json_data = f.read()
crawled_movies = json.loads(json_data)
preprocessor = Preprocessor(crawled_movies)
prep_movies = preprocessor.preprocess()
index = Index(prep_movies)

index.check_add_remove_is_correct()

for item in ['documents', 'stars', 'genres', 'summaries']:
    print(item, ": ")
    index.check_if_indexing_is_good(item)
    print('****************************')

for item in ['documents', 'stars', 'genres', 'summaries']:
    path = 'index/' + item + '_index.json'
    index.store_index(path, item)

for item in ['documents', 'stars', 'genres', 'summaries']:
    path = 'index/' + item + '_index.json'
    index.load_index(path)
    print(index.check_if_index_loaded_correctly(item, index.index[item]))
    print('****************************')