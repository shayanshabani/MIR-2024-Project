import time
import os
import json
from enum import Enum
import copy


class Indexes(Enum):
    DOCUMENTS = 'documents'
    STARS = 'stars'
    GENRES = 'genres'
    SUMMARIES = 'summaries'


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

        # TODO
        pass

    def index_stars(self):
        """
        Index the documents based on the stars.

        Returns
        ----------
        dict
            The index of the documents based on the stars.
        """

        #         TODO
        pass

    def index_genres(self):
        """
        Index the documents based on the genres.

        Returns
        ----------
        dict
            The index of the documents based on the genres.
        """

        #         TODO
        pass

    def index_summaries(self):
        """
        Index the documents based on the summaries (not first_page_summary).

        Returns
        ----------
        dict
            The index of the documents based on the summaries.
        """

        #         TODO
        pass

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
        dict
            posting list
        """

        #         TODO
        pass

    def add_document_to_index(self, document: dict):
        """
        Add a document to all the indexes

        Parameters
        ----------
        document : dict
            Document to add to all the indexes
        """

        #         TODO
        pass

    def remove_document_from_index(self, document_id: str):
        """
        Remove a document from all the indexes

        Parameters
        ----------
        document_id : str
            ID of the document to remove from all the indexes
        """

        #         TODO
        pass

    def check_add_remove_is_correct(self):
        """
        Check if the add and remove is correct
        """

        dummy_document = {
            'id': '100',
            'stars': ['Tim', 'Tom'],
            'genres': ['Drama', 'Comedy'],
            'summaries': ['good']
        }

        index_before_add = copy.deepcopy(self.index)
        self.add_document_to_index(dummy_document)
        index_after_add = copy.deepcopy(self.index)

        if index_after_add[Indexes.DOCUMENTS.value]['100'] != dummy_document:
            print('Add is incorrect, document')
            return

        if (set(index_after_add[Indexes.STARS.value]['Tim']).difference(
                set(index_before_add[Indexes.STARS.value]['Tim']))
                != {dummy_document['id']}):
            print('Add is incorrect, Tim')
            return

        if (set(index_after_add[Indexes.STARS.value]['Tom']).difference(
                set(index_before_add[Indexes.STARS.value]['Tom']))
                != {dummy_document['id']}):
            print('Add is incorrect, Tom')
            return
        if (set(index_after_add[Indexes.GENRES.value]['Drama']).difference(
                set(index_before_add[Indexes.GENRES.value]['Drama']))
                != {dummy_document['id']}):
            print('Add is incorrect, Drama')
            return

        if (set(index_after_add[Indexes.GENRES.value]['Comedy']).difference(
                set(index_before_add[Indexes.GENRES.value]['Comedy']))
                != {dummy_document['id']}):
            print('Add is incorrect, Comedy')
            return

        if (set(index_after_add[Indexes.SUMMARIES.value]['good']).difference(
                set(index_before_add[Indexes.SUMMARIES.value]['good']))
                != {dummy_document['id']}):
            print('Add is incorrect, good')
            return

        print('Add is correct')

        self.remove_document_from_index('100')
        index_after_remove = copy.deepcopy(self.index)

        if index_after_remove == index_before_add:
            print('Remove is correct')
        else:
            print('Remove is incorrect')

    def store_index(self, path: str, index_type: str):
        """
        Stores the index in a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to store the file
        index_type: str
            type of index we want to store (documents, stars, genres, summaries)
        """

        #         TODO
        pass

    def load_index(self, path: str):
        """
        Loads the index from a file (such as a JSON file)

        Parameters
        ----------
        path : str
            Path to load the file
        """

        #         TODO
        pass

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

    def check_if_indexing_is_good(self, index_type: str, check_word: str = 'good'):
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
        for document in self.preprocessed_documents:
            if index_type not in document or document[index_type] is None:
                continue

            for field in document[index_type]:
                if check_word in field:
                    docs.append(document['id'])
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

        print("Brute force time: ", brute_force_time)
        print("Implemented time: ", implemented_time)

        print(docs)
        print(posting_list)
        if set(docs).issubset(set(posting_list)):
            print("Indexing is correct")

            if implemented_time < brute_force_time:
                print("Indexing is good")
                return True
            else:
                print("Indexing is bad")
                return False
        else:
            print("Indexing is wrong")
            return False


# TODO: Run the class with needed parameters, then run check methods and finally report the results of check methods
