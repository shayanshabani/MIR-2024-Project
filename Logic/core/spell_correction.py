class SpellCorrection:
    def __init__(self, all_documents):
        """
        Initialize the SpellCorrection

        Parameters
        ----------
        all_documents : list of str
            The input documents.
        """
        self.all_shingled_words, self.word_counter = self.shingling_and_counting(all_documents)

    def shingle_word(self, word, k=2):
        """
        Convert a word into a set of shingles.

        Parameters
        ----------
        word : str
            The input word.
        k : int
            The size of each shingle.

        Returns
        -------
        set
            A set of shingles.
        """
        shingles = set()

        # TODO: Create shingle here

        for i in range(len(word) - 1):
            shingle = ''
            for j in range(k):
                shingle += word[i + j]
            shingles.add(shingle)
        return shingles
    
    def jaccard_score(self, first_set, second_set):
        """
        Calculate jaccard score.

        Parameters
        ----------
        first_set : set
            First set of shingles.
        second_set : set
            Second set of shingles.

        Returns
        -------
        float
            Jaccard score.
        """

        # TODO: Calculate jaccard score here.
        union = len(first_set.union(second_set))
        intersection = len(first_set.intersection(second_set))
        if union != 0:
            jaccard_score = intersection / union
        else:
            jaccard_score = 0
        return jaccard_score

    def shingling_and_counting(self, all_documents):
        """
        Shingle all words of the corpus and count TF of each word.

        Parameters
        ----------
        all_documents : list of str
            The input documents.

        Returns
        -------
        all_shingled_words : dict
            A dictionary from words to their shingle sets.
        word_counter : dict
            A dictionary from words to their TFs.
        """
        all_shingled_words = dict()
        word_counter = dict()

        # TODO: Create shingled words dictionary and word counter dictionary here.

        for document in all_documents:
            words = document.split()
            for word in words:
                shingle_set = self.shingle_word(word)
                if word not in all_shingled_words:
                    all_shingled_words[word] = shingle_set
                if word not in word_counter:
                    word_counter[word] = 0
                word_counter[word] += 1

        return all_shingled_words, word_counter
    
    def find_nearest_words(self, word):
        """
        Find correct form of a misspelled word.

        Parameters
        ----------
        word : stf
            The misspelled word.

        Returns
        -------
        list of str
            5 nearest words.
        """
        top5_candidates = list()

        # TODO: Find 5 nearest candidates here.
        word_shingles = self.shingle_word(word)

        similarities = {}
        for candidate_word, candidate_shingles in self.all_shingled_words.items():
            similarities[candidate_word] = self.jaccard_score(word_shingles, candidate_shingles)

        nearest_words = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:5]

        for word, _ in nearest_words:
            top5_candidates.append(word)

        return top5_candidates

    def count_embedding(self, top5_candidates, word):
        """
        Calculate total scores.

        Parameters
        ----------
        top5_candidates : list of str
            List of 5 nearest words.
        word : str
            The misspelled word.

        Returns
        -------
        str
            The best candidate word.
        """
        word_shingles = self.all_shingled_words[word]
        normalized_tf = {}
        max_tf = max(self.word_counter[candidate] for candidate in top5_candidates)
        for candidate in top5_candidates:
            candidate_shingles = self.all_shingled_words[candidate]
            candidate_tf = self.word_counter[candidate]
            candidate_jaccard = self.jaccard_score(word_shingles, candidate_shingles)
            normalized_tf[candidate] = (candidate_jaccard * candidate_tf) / max_tf
        best_candidate = max(normalized_tf, key=normalized_tf.get)
        return best_candidate

    
    def spell_check(self, query):
        """
        Find correct form of a misspelled query.

        Parameters
        ----------
        query : stf
            The misspelled query.

        Returns
        -------
        str
            Correct form of the query.
        """
        final_result = ""
        
        # TODO: Do spell correction here.
        words = query.split()
        for word in words:
            if word in self.word_counter:
                final_result += word + ' '
            else:
                top5_candidates = self.find_nearest_words(word)
                best_candidate = self.count_embedding(top5_candidates, word)
                final_result += best_candidate + ' '
        final_result = final_result.strip()
        return final_result