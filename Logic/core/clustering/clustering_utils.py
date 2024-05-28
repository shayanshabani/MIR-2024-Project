import matplotlib.pyplot as plt
import numpy as np
import random
import operator
import wandb

from typing import List, Tuple
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from collections import Counter
from Logic.core.clustering.clustering_metrics import *
import dimension_reduction

class ClusteringUtils:

    def cluster_kmeans(self, emb_vecs: List, n_clusters: int, max_iter: int = 100) -> Tuple[List, List]:
        """
        Clusters input vectors using the K-means method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List]
            Two lists:
            1. A list containing the cluster centers.
            2. A list containing the cluster index for each input vector.
        """
        # indices = random.sample(range(len(emb_vecs)), n_clusters)
        indices = random.sample(range(len(emb_vecs)), 1)

        for i in range(n_clusters - 1):
            distances = []
            for j in range(len(emb_vecs)):
                distance = 0
                for index in indices:
                    distance += np.linalg.norm(emb_vecs[j] - emb_vecs[index])
                distance /= len(indices)
                distances.append(distance)
            indices.append(np.argmax(distances))

        centroids = [emb_vecs[idx] for idx in indices]
        assignments = [0] * len(emb_vecs)

        for iteration in range(max_iter):
            updated_assignments = []

            for vector in emb_vecs:
                dists = [np.linalg.norm(vector - centroid) for centroid in centroids]
                closest_cluster = dists.index(min(dists))
                updated_assignments.append(closest_cluster)

            if updated_assignments == assignments:
                break
            assignments = updated_assignments

            for cluster_idx in range(n_clusters):
                cluster_vectors = [emb_vecs[idx] for idx, assignment in enumerate(assignments) if assignment == cluster_idx]
                if cluster_vectors:
                    centroids[cluster_idx] = np.mean(cluster_vectors, axis=0)
        return centroids, assignments

    def get_most_frequent_words(self, documents: List[str], top_n: int = 10) -> List[Tuple[str, int]]:
        """
        Finds the most frequent words in a list of documents.

        Parameters
        -----------
        documents: List[str]
            A list of documents, where each document is a string representing a list of words.
        top_n: int, optional
            The number of most frequent words to return. Default is 10.

        Returns
        --------
        List[Tuple[str, int]]
            A list of tuples, where each tuple contains a word and its frequency, sorted in descending order of frequency.
        """
        return Counter(word for doc in documents for word in doc.split()).most_common(top_n)


    def cluster_kmeans_WCSS(self, emb_vecs: List, n_clusters: int) -> Tuple[List, List, float]:
        """ This function performs K-means clustering on a list of input vectors and calculates the Within-Cluster Sum of Squares (WCSS) for the resulting clusters.

        This function implements the K-means algorithm and returns the cluster centroids, cluster assignments for each input vector, and the WCSS value.

        The WCSS is a measure of the compactness of the clustering, and it is calculated as the sum of squared distances between each data point and its assigned cluster centroid. A lower WCSS value indicates that the data points are closer to their respective cluster centroids, suggesting a more compact and well-defined clustering.

        The K-means algorithm works by iteratively updating the cluster centroids and reassigning data points to the closest centroid until convergence or a maximum number of iterations is reached. This function uses a random initialization of the centroids and runs the algorithm for a maximum of 100 iterations.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.
        n_clusters: int
            The number of clusters to form.

        Returns
        --------
        Tuple[List, List, float]
            Three elements:
            1) A list containing the cluster centers.
            2) A list containing the cluster index for each input vector.
            3) The Within-Cluster Sum of Squares (WCSS) value for the clustering.
        """
        centroids, assignments = self.cluster_kmeans(emb_vecs, n_clusters)
        within_cluster_sum_square = sum(np.linalg.norm(centroids[assignments[idx]] - vector) ** 2 for idx, vector in enumerate(emb_vecs))
        return centroids, assignments, within_cluster_sum_square

    def cluster_hierarchical_single(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with single linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='single')
        cluster_indices = agglomerative_clustering.fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_complete(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with complete linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='complete')
        cluster_indices = agglomerative_clustering.fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_average(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with average linkage.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='average')
        cluster_indices = agglomerative_clustering.fit_predict(emb_vecs)
        return cluster_indices

    def cluster_hierarchical_ward(self, emb_vecs: List) -> List:
        """
        Clusters input vectors using the hierarchical clustering method with Ward's method.

        Parameters
        -----------
        emb_vecs: List
            A list of vectors to be clustered.

        Returns
        --------
        List
            A list containing the cluster index for each input vector.
        """
        agglomerative_clustering = AgglomerativeClustering(n_clusters=2, linkage='ward')
        cluster_indices = agglomerative_clustering.fit_predict(emb_vecs)
        return cluster_indices

    def visualize_kmeans_clustering_wandb(self, data, n_clusters, project_name, run_name):
        """ This function performs K-means clustering on the input data and visualizes the resulting clusters by logging a scatter plot to Weights & Biases (wandb).

        This function applies the K-means algorithm to the input data and generates a scatter plot where each data point is colored according to its assigned cluster.
        For visualization use convert_to_2d_tsne to make your scatter plot 2d and visualizable.
        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform K-means clustering on the input data with the specified number of clusters.
        3. Obtain the cluster labels for each data point from the K-means model.
        4. Create a scatter plot of the data, coloring each point according to its cluster label.
        5. Log the scatter plot as an image to the wandb run, allowing visualization of the clustering results.
        6. Close the plot display window to conserve system resources (optional).

        Parameters
        -----------
        data: np.ndarray
            The input data to perform K-means clustering on.
        n_clusters: int
            The number of clusters to form during the K-means clustering process.
        project_name: str
            The name of the wandb project to log the clustering visualization.
        run_name: str
            The name of the wandb run to log the clustering visualization.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Perform K-means clustering
        # TODO
        centroids, labels = self.cluster_kmeans(list(data), n_clusters)
        dm = dimension_reduction.DimensionReduction()
        extended_data = np.vstack((data, centroids))
        tsne_result = dm.convert_to_2d_tsne(extended_data)
        data_2d = tsne_result[:-n_clusters]
        centroids_2d = tsne_result[-n_clusters:]
        # Plot the clusters
        # TODO
        figure, ax = plt.subplots()
        ax.scatter(data_2d[:, 0], data_2d[:, 1], c=labels, cmap='plasma')
        ax.scatter(centroids_2d[:, 0], centroids_2d[:, 1], c='red', marker='x', s=100)
        ax.set_title(f'{n_clusters}-Means Clustering')
        plt.show()

        # Log the plot to wandb
        # TODO
        wandb.log({f"{n_clusters}-Means Clustering": wandb.Image(figure)})
        # Close the plot display window if needed (optional)
        # TODO



    def wandb_plot_hierarchical_clustering_dendrogram(self, data, project_name, linkage_method, run_name):
        """ This function performs hierarchical clustering on the provided data and generates a dendrogram plot, which is then logged to Weights & Biases (wandb).

        The dendrogram is a tree-like diagram that visualizes the hierarchical clustering process. It shows how the data points (or clusters) are progressively merged into larger clusters based on their similarity or distance.

        The function performs the following steps:
        1. Initialize a new wandb run with the provided project and run names.
        2. Perform hierarchical clustering on the input data using the specified linkage method.
        3. Create a linkage matrix, which represents the merging of clusters at each step of the hierarchical clustering process.
        4. Generate a dendrogram plot using the linkage matrix.
        5. Log the dendrogram plot as an image to the wandb run.
        6. Close the plot display window to conserve system resources.

        Parameters
        -----------
        data: np.ndarray
            The input data to perform hierarchical clustering on.
        linkage_method: str
            The linkage method for hierarchical clustering. It can be one of the following: "average", "ward", "complete", or "single".
        project_name: str
            The name of the wandb project to log the dendrogram plot.
        run_name: str
            The name of the wandb run to log the dendrogram plot.

        Returns
        --------
        None
        """
        run = wandb.init(project=project_name, name=run_name)
        # Perform hierarchical clustering
        # TODO
        linkage_matrix = linkage(data, method=linkage_method)
        # Create linkage matrix for dendrogram
        # TODO
        f = plt.figure(figsize=(10, 7))
        dendrogram(linkage_matrix)
        plt.show()
        wandb.log({"dendrogram Method": wandb.Image(f)})

    def plot_kmeans_cluster_scores(self, embeddings: List, true_labels: List, k_values: List[int], project_name=None,
                                   run_name=None):
        """ This function, using implemented metrics in clustering_metrics, calculates and plots both purity scores and silhouette scores for various numbers of clusters.
        Then using wandb plots the respective scores (each with a different color) for each k value.

        Parameters
        -----------
        embeddings : List
            A list of vectors representing the data points.
        true_labels : List
            A list of ground truth labels for each data point.
        k_values : List[int]
            A list containing the various values of 'k' (number of clusters) for which the scores will be calculated.
            Default is range(2, 9), which means it will calculate scores for k values from 2 to 8.
        project_name : str
            Your wandb project name. If None, the plot will not be logged to wandb. Default is None.
        run_name : str
            Your wandb run name. If None, the plot will not be logged to wandb. Default is None.

        Returns
        --------
        None
        """
        silhouette_scores = []
        purity_scores = []
        # Calculating Silhouette Scores and Purity Scores for different values of k
        cluster_metrics = ClusteringMetrics()
        for k in k_values:
            # TODO
            _, assignments = self.cluster_kmeans(embeddings, k)
            # Using implemented metrics in clustering_metrics, get the score for each k in k-means clustering
            # and visualize it.
            # TODO
            silhouette_scores.append(cluster_metrics.silhouette_score(embeddings, assignments))
            purity_scores.append(cluster_metrics.purity_score(true_labels, assignments))

        # Plotting the scores
        # TODO
        figure, (silhouette_ax, purity_ax) = plt.subplots(nrows=2, ncols=1, figsize=(10, 8))

        silhouette_ax.plot(k_values, silhouette_scores, label='Silhouette Score', color='blue')
        silhouette_ax.set_xlabel('Number of Clusters (k)')
        silhouette_ax.set_ylabel('Silhouette Score', color='blue')
        silhouette_ax.legend(loc='upper right')
        silhouette_ax.set_title('Silhouette Score vs. Number of Clusters')

        purity_ax.plot(k_values, purity_scores, label='Purity Score', color='red')
        purity_ax.set_xlabel('Number of Clusters (k)')
        purity_ax.set_ylabel('Purity Score', color='red')
        purity_ax.legend(loc='upper right')
        purity_ax.set_title('Purity Score vs. Number of Clusters')

        plt.tight_layout()

        plt.show()

        # Logging the plot to wandb
        if project_name and run_name:
            import wandb
            run = wandb.init(project=project_name, name=run_name)
            wandb.log({"Cluster Scores": wandb.Image(figure)})

    def visualize_elbow_method_wcss(self, embeddings: List, k_values: List[int], project_name: str, run_name: str):
        """ This function implements the elbow method to determine the optimal number of clusters for K-means clustering based on the Within-Cluster Sum of Squares (WCSS).

        The elbow method is a heuristic used to determine the optimal number of clusters in K-means clustering. It involves plotting the WCSS values for different values of K (number of clusters) and finding the "elbow" point in the curve, where the marginal improvement in WCSS starts to diminish. This point is considered as the optimal number of clusters.

        The function performs the following steps:
        1. Iterate over the specified range of K values.
        2. For each K value, perform K-means clustering using the `cluster_kmeans_WCSS` function and store the resulting WCSS value.
        3. Create a line plot of WCSS values against the number of clusters (K).
        4. Log the plot to Weights & Biases (wandb) for visualization and tracking.

        Parameters
        -----------
        embeddings: List
            A list of vectors representing the data points to be clustered.
        k_values: List[int]
            A list of K values (number of clusters) to explore for the elbow method.
        project_name: str
            The name of the wandb project to log the elbow method plot.
        run_name: str
            The name of the wandb run to log the elbow method plot.

        Returns
        --------
        None
        """
        # Initialize wandb
        run = wandb.init(project=project_name, name=run_name)

        # Compute WCSS values for different K values
        wcss_values = []
        for k in k_values:
            # TODO
            _, _, within_cluster_sum_square = self.cluster_kmeans_WCSS(embeddings, k)
            wcss_values.append(within_cluster_sum_square)

        # Plot the elbow method
        # TODO
        plt.plot(k_values, wcss_values, marker='o')
        plt.show()
        # Log the plot to wandb
        wandb.log({"Elbow Method": wandb.Image(plt)})

        plt.close()