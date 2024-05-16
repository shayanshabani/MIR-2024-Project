class Snippet:
    def __init__(self, number_of_words_on_each_side=5):
        """
        Initialize the Snippet

        Parameters
        ----------
        number_of_words_on_each_side : int
            The number of words on each side of the query word in the doc to be presented in the snippet.
        """
        self.number_of_words_on_each_side = number_of_words_on_each_side

    def remove_stop_words_from_query(self, query):
        """
        Remove stop words from the input string.

        Parameters
        ----------
        query : str
            The query that you need to delete stop words from.

        Returns
        -------
        str
            The query without stop words.
        """

        # TODO: remove stop words from the query.
        words = query.split()
        stop_words = self.load_stopwords()
        filtered_words = [word for word in words if word not in stop_words]
        filtered_words = ' '.join(filtered_words)
        return filtered_words

    def find_snippet(self, doc, query):
        """
        Find snippet in a doc based on a query.

        Parameters
        ----------
        doc : str
            The retrieved doc which the snippet should be extracted from that.
        query : str
            The query which the snippet should be extracted based on that.

        Returns
        -------
        final_snippet : str
            The final extracted snippet. IMPORTANT: The keyword should be wrapped by *** on both sides.
            For example: Sahwshank ***redemption*** is one of ... (for query: redemption)
        not_exist_words : list
            Words in the query which don't exist in the doc.
        """
        final_snippet = ""
        not_exist_words = []

        # TODO: Extract snippet and the tokens which are not present in the doc.

        query_without_stopwords = self.remove_stop_words_from_query(query)
        query_tokens = query_without_stopwords.split()
        doc_tokens = doc.split()
        not_exist_words = [token for token in query_tokens if token not in doc_tokens]
        snippet_list = []

        for token in query_tokens:
            if token in doc_tokens:
                occurrences = [i for i, x in enumerate(doc_tokens) if x == token]
                for occurrence in occurrences:
                    start_index = max(0, occurrence - self.number_of_words_on_each_side)
                    end_index = min(len(doc_tokens), occurrence + self.number_of_words_on_each_side + 1)
                    snippet = ' '.join(doc_tokens[start_index:end_index])
                    snippet_list.append(snippet)

        final_snippet = ' ... '.join(snippet_list)
        for token in query_tokens:
            final_snippet = final_snippet.replace(token, '***{}***'.format(token))

        return final_snippet, not_exist_words

    def load_stopwords(self):
        """
        Load stopwords.

        Returns
        ----------
        set
            A set containing the stopwords.
        """
        stopwords = ['this', 'that', 'about', 'whom', 'being', 'where', 'why', 'had', 'should', 'each']
        return stopwords