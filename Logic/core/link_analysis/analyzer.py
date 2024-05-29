# from .graph import LinkGraph
# from ..indexer.indexes_enum import Indexes
# from ..indexer.index_reader import Index_reader
import json

from Logic.core.link_analysis.graph import LinkGraph
from Logic.core.indexer.indexes_enum import Indexes
from Logic.core.indexer.index_reader import Index_reader

class LinkAnalyzer:
    def __init__(self, root_set):
        """
        Initialize the Link Analyzer attributes:

        Parameters
        ----------
        root_set: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "title": string of movie title
            "stars": A list of movie star names
        """
        self.root_set = root_set
        self.graph = LinkGraph()
        self.hubs = []
        self.authorities = []
        self.initiate_params()

    def initiate_params(self):
        """
        Initialize links graph, hubs list and authorities list based of root set

        Parameters
        ----------
        This function has no parameters. You can use self to get or change attributes
        """
        for movie in self.root_set:
            #TODO
            for star in movie['stars']:
                if star not in self.authorities:
                    self.authorities.append(star)
                    self.graph.add_node(star)
                    self.graph.add_edge(movie['id'], star)

            if movie['id'] not in self.hubs:
                self.hubs.append(movie['id'])
                self.graph.add_node(movie['id'])




    def expand_graph(self, corpus):
        """
        expand hubs, authorities and graph using given corpus

        Parameters
        ----------
        corpus: list
            A list of movie dictionaries with the following keys:
            "id": A unique ID for the movie
            "stars": A list of movie star names

        Note
        ---------
        To build the base set, we need to add the hubs and authorities that are inside the corpus
        and refer to the nodes in the root set to the graph and to the list of hubs and authorities.
        """
        for movie in corpus:
            if movie['id'] not in self.hubs and movie['stars'] is not None:
                for star in movie['stars']:
                    if star in self.authorities:
                        self.graph.add_node(movie['id'])
                        if movie['id'] not in self.hubs:
                            self.hubs.append(movie['id'])

                        self.graph.add_edge(movie['id'], star)


    def hits(self, num_iteration=5, max_result=10):
        """
        Return the top movies and actors using the Hits algorithm

        Parameters
        ----------
        num_iteration: int
            Number of algorithm execution iterations
        max_result: int
            The maximum number of results to return. If None, all results are returned.

        Returns
        -------
        list
            List of names of 10 actors with the most scores obtained by Hits algorithm in descending order
        list
            List of names of 10 movies with the most scores obtained by Hits algorithm in descending order
        """

        #TODO

        auth_scores = {node: 1.0 for node in self.authorities}
        hub_scores = {node: 1.0 for node in self.hubs}

        for _ in range(num_iteration):
            for node in self.authorities:
                auth_scores[node] = sum(hub_scores[hub] for hub in self.graph.get_predecessors(node))

            for node in self.hubs:
                hub_scores[node] = sum(auth_scores[auth] for auth in self.graph.get_successors(node))

            auth_norm = sum(auth_scores.values())
            hub_norm = sum(hub_scores.values())

            for node in self.authorities:
                auth_scores[node] /= auth_norm
            for node in self.hubs:
                hub_scores[node] /= hub_norm

        a_s = sorted(auth_scores.items(), key=lambda item: item[1], reverse=True)[:max_result]
        h_s = sorted(hub_scores.items(), key=lambda item: item[1], reverse=True)[:max_result]

        return a_s, h_s

if __name__ == "__main__":
    # You can use this section to run and test the results of your link analyzer
    corpus = []    # TODO: it should be your crawled data
    root_set = []   # TODO: it should be a subset of your corpus
    path = '../indexer/index/documents_index.json'
    with open(path, 'r') as f:
        json_data = f.read()
    document_index = json.loads(json_data)
    corpus = list(document_index.values())
    root_set = corpus[:15]
    analyzer = LinkAnalyzer(root_set=root_set)
    analyzer.expand_graph(corpus=corpus)
    actors, movies = analyzer.hits(max_result=5)
    print("Top Actors:")
    print(*actors, sep=' - ')
    print("Top Movies:")
    print(*movies, sep=' - ')
