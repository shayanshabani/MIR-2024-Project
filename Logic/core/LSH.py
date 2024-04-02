import numpy as np
import itertools
import random


class MinHashLSH:
    def __init__(self, documents, num_hashes):
        """
        Initialize the MinHashLSH

        Parameters
        ----------
        documents : list of str
            The input documents for similarity analysis.
        num_hashes : int
            Number of hashes for mini-hashing.
        """
        self.documents = documents
        self.num_hashes = num_hashes
        self.universal_shingles = []
        self.characteristic_matrix = None
        self.signature_matrix = None

    def shingle_document(self, document, k=2):
        """
        Convert a document into a set of shingles.

        Parameters
        ----------
        document : str
            The input document.
        k : int
            The size of each shingle.

        Returns
        ----------
        set
            A set of shingles.
        """
        words = document.split()
        shingles = set()

        for i in range(len(words) - 1):
            shingle = ''
            for j in range(k):
                shingle += words[i + j]
                shingle += ' '
            shingle = shingle.strip()
            shingles.add(shingle)

        return shingles

    def build_characteristic_matrix(self):
        """
        Build the characteristic matrix representing the presence of shingles in documents.

        Returns
        ----------
        numpy.ndarray
            The binary characteristic matrix.
        """
        # TODO
        shingles_list = []
        for document in self.documents:
            shingles_list.append(self.shingle_document(document, 2))
        universal_shingles_set = set().union(*shingles_list)
        universal_shingles = list(universal_shingles_set)
        self.universal_shingles = universal_shingles
        size_of_rows = len(universal_shingles)
        size_of_columns = len(self.documents)
        characteristic_matrix = np.ndarray(shape=(size_of_rows, size_of_columns), dtype=np.int)
        for i in range(size_of_columns):
            for j in range(size_of_rows):
                if universal_shingles[j] in shingles_list[i]:
                    characteristic_matrix[j][i] = 1
                else:
                    characteristic_matrix[j][i] = 0
        self.characteristic_matrix = characteristic_matrix
        return characteristic_matrix

    def min_hash_signature(self):
        """
        Perform Min-Hashing to generate hash signatures for documents.

        Returns
        ----------
        numpy.ndarray
            The Min-Hash signatures matrix.
        """
        # TODO
        index_list = []
        for i in range(len(self.universal_shingles)):
            index_list.append(i)
        num_of_shuffle = 100
        signature_matrix = np.ndarray(shape=(num_of_shuffle, len(self.documents)), dtype=np.int)
        self.signature_matrix = signature_matrix
        for i in range(num_of_shuffle):
            random.shuffle(index_list)
            for j in range(len(self.documents)):
                for k in range(len(index_list)):
                    if self.characteristic_matrix[index_list.index(k)][j] == 1:
                        signature_matrix[i][j] = k
                        break

        return signature_matrix

    def lsh_buckets(self, signature, bands=10, rows_per_band=10):
        """
        Group documents into Locality-Sensitive Hashing (LSH) buckets based on Min-Hash signatures.

        Parameters
        ----------
        signature : numpy.ndarray
            Min-Hash signatures for documents.
        bands : int
            Number of bands for LSH.
        rows_per_band : int
            Number of rows per band.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        num_of_bands = signature.shape[0] / bands
        buckets = {}
        # for band_idx in range(num_of_bands):
        #     start_row = band_idx * rows_per_band
        #     end_row = (band_idx + 1) * rows_per_band
        #
        #     band_hashes = [hash(tuple(row)) for row in signature[start_row:end_row]]
        #
        #     for idx, hash_key in enumerate(band_hashes):
        #         bucket_id = (band_idx, hash_key)
        #         if bucket_id not in buckets:
        #             buckets[bucket_id] = []
        #         buckets[bucket_id].append(start_row + idx)
        return buckets

    def perform_lsh(self):
        """
        Perform the entire Locality-Sensitive Hashing (LSH) process.

        Returns
        ----------
        dict
            A dictionary mapping bucket IDs to lists of document indices.
        """
        # TODO
        return

    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score for two sets.

        Parameters
        ----------
        first_set : set
            Set of first shingled document.
        second_set : set
            Set of second shingled document.

        Returns
        ----------
        float
            Jaccard score.
        """
        # TODO
        union = first_set.union(second_set)
        intersection = first_set.intersection(second_set)
        jaccaurd_score = intersection / union
        return jaccaurd_score

    def jaccard_similarity_test(self, buckets, all_documents):
        """
        Test your near duplicate detection code based on jaccard similarity.

        Parameters
        ----------
        buckets : dict
            A dictionary mapping bucket IDs to lists of document indices.
        all_documents : list
            The input documents for similarity analysis.
        """
        correct_near_duplicates = 0
        all_near_duplicates = 0

        for bucket_id in buckets.keys():
            docs_in_this_bucket = buckets[bucket_id]
            unique_doc_ids = set(docs_in_this_bucket)
            if len(unique_doc_ids) > 1:
                combinations = list(itertools.combinations(unique_doc_ids, 2))
                for comb in combinations:
                    all_near_duplicates += 1

                    first_doc_id = comb[0]
                    second_doc_id = comb[1]

                    first_shingled_doc = self.shingle_document(all_documents[first_doc_id], 2)
                    second_shingled_doc = self.shingle_document(all_documents[second_doc_id], 2)

                    near_duplicated_jaccard_score = self.jaccard_score(first_shingled_doc, second_shingled_doc)
                    current_score = 0

                    for _ in range(5):
                        random_doc_id = first_doc_id
                        while random_doc_id == first_doc_id or random_doc_id == second_doc_id:
                            random_doc_id = random.randint(0, len(all_documents) - 1)
                        random_shingled_doc = self.shingle_document(all_documents[random_doc_id], 2)

                        random_jaccard_score = self.jaccard_score(first_shingled_doc, random_shingled_doc)

                        if near_duplicated_jaccard_score > random_jaccard_score:
                            current_score += 1

                    if current_score == 5:
                        correct_near_duplicates += 1

        # a good score is around 0.8
        print("your final score in near duplicate detection:", correct_near_duplicates / all_near_duplicates)
