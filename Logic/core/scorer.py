import numpy as np

class Scorer:    
    def __init__(self, index, number_of_documents):
        """
        Initializes the Scorer.

        Parameters
        ----------
        index : dict
            The index to score the documents with.
        number_of_documents : int
            The number of documents in the index.
        """

        self.index = index
        self.idf = {}
        self.N = number_of_documents

    def get_list_of_documents(self,query):
        """
        Returns a list of documents that contain at least one of the terms in the query.

        Parameters
        ----------
        query: List[str]
            The query to be scored

        Returns
        -------
        list
            A list of documents that contain at least one of the terms in the query.
        
        Note
        ---------
            The current approach is not optimal but we use it due to the indexing structure of the dict we're using.
            If we had pairs of (document_id, tf) sorted by document_id, we could improve this.
                We could initialize a list of pointers, each pointing to the first element of each list.
                Then, we could iterate through the lists in parallel.
            
        """
        list_of_documents = []
        for term in query:
            if term in self.index.keys():
                list_of_documents.extend(self.index[term].keys())
        return list(set(list_of_documents))
    
    def get_idf(self, term):
        """
        Returns the inverse document frequency of a term.

        Parameters
        ----------
        term : str
            The term to get the inverse document frequency for.

        Returns
        -------
        float
            The inverse document frequency of the term.
        
        Note
        -------
            It was better to store dfs in a separate dict in preprocessing.
        """
        idf = self.idf.get(term, None)
        if idf is None:
            # TODO
            df = 0
            if term in self.index:
                df = len(self.index[term])
            if df != 0:
                idf = np.log10(self.N) / df
            else:
                idf = 0
        return idf
    
    def get_query_tfs(self, query):
        """
        Returns the term frequencies of the terms in the query.

        Parameters
        ----------
        query : List[str]
            The query to get the term frequencies for.

        Returns
        -------
        dict
            A dictionary of the term frequencies of the terms in the query.
        """
        
        #TODO
        term_frequencies = {}
        for i in range(len(query)):
            term = query[i]
            if term not in term_frequencies:
                term_frequencies[term] = 0
            term_frequencies[term] += 1
        return term_frequencies

    def compute_scores_with_vector_space_model(self, query, method):
        """
        compute scores with vector space model

        Parameters
        ----------
        query: List[str]
            The query to be scored
        method : str ((n|l)(n|t)(n|c).(n|l)(n|t)(n|c))
            The method to use for searching.

        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        document_scores = {}
        query_tfs = self.get_query_tfs(query)

        documents = self.get_list_of_documents(query)
        for document in documents:
            document_scores[document['id']] = self.get_vector_space_model_score(
                query, query_tfs, document['id'], method[:3], method[4:]
            )
        return document_scores



    def get_vector_space_model_score(self, query, query_tfs, document_id, document_method, query_method):
        """
        Returns the Vector Space Model score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        query_tfs : dict
            The term frequencies of the terms in the query.
        document_id : str
            The document to calculate the score for.
        document_method : str (n|l)(n|t)(n|c)
            The method to use for the document.
        query_method : str (n|l)(n|t)(n|c)
            The method to use for the query.

        Returns
        -------
        float
            The Vector Space Model score of the document for the query.
        """

        #TODO
        distinct_terms = list(set(query))
        query_weights = []
        doc_weights = []
        for term in distinct_terms:
            # compute weight for term
            query_term_frequency = query_tfs[term]
            if query_method[0] == 'l':
                query_term_frequency = 1 + np.log10(query_term_frequency)
            query_document_frequency = 1
            if query_method[1] == 't':
                query_document_frequency = self.get_idf(term)
            # compute weight for doc
            doc_term_frequency = self.index[term][document_id]
            if document_method[0] == 'l':
                doc_term_frequency = 1 + np.log10(doc_term_frequency)
            doc_document_frequency = 1
            if document_method[1] == 't':
                doc_document_frequency = self.get_idf(term)

            query_weights.append(query_term_frequency * query_document_frequency)
            doc_weights.append(doc_term_frequency * doc_document_frequency)

        # normalize query weights
        query_normalization = 1
        if query_method[2] == 'c':
            total = 0
            for weight in query_weights:
                total += weight ** 2
            query_normalization = np.sqrt(total)
        query_score = [weight / query_normalization for weight in query_weights]
        # normalize doc weights
        doc_normalization = 1
        if document_method[2] == 'c':
            total = 0
            for weight in query_weights:
                total += weight ** 2
            doc_normalization = np.sqrt(total)
        doc_score = [weight / doc_normalization for weight in doc_weights]
        final_score = [doc_weight * query_weight for (query_weight, doc_weight) in zip(query_score, doc_score)]
        score = 0
        for final in final_score:
            score += final
        return score


    def compute_scores_with_okapi_bm25(self, query, average_document_field_length, document_lengths):
        """
        compute scores with okapi bm25

        Parameters
        ----------
        query: List[str]
            The query to be scored
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.
        
        Returns
        -------
        dict
            A dictionary of the document IDs and their scores.
        """

        # TODO
        pass

    def get_okapi_bm25_score(self, query, document_id, average_document_field_length, document_lengths):
        """
        Returns the Okapi BM25 score of a document for a query.

        Parameters
        ----------
        query: List[str]
            The query to be scored
        document_id : str
            The document to calculate the score for.
        average_document_field_length : float
            The average length of the documents in the index.
        document_lengths : dict
            A dictionary of the document lengths. The keys are the document IDs, and the values are
            the document's length in that field.

        Returns
        -------
        float
            The Okapi BM25 score of the document for the query.
        """

        # TODO
        pass