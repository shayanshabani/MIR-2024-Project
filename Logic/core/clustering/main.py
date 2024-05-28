import numpy as np
import os
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import wandb
from Logic.core.word_embedding.fasttext_data_loader import FastTextDataLoader
from Logic.core.word_embedding.fasttext_model import FastText
from Logic.core.clustering.dimension_reduction import DimensionReduction
from Logic.core.clustering.clustering_metrics import ClusteringMetrics
from Logic.core.clustering.clustering_utils import ClusteringUtils

# Main Function: Clustering Tasks

# 0. Embedding Extraction
# TODO: Using the previous preprocessor and fasttext model, collect all the embeddings of our data and store them.
wandb.login(key="de3b097f1c3c879ce1cb5773775f2be151463c16")
ft_model = FastText()
ft_model.prepare(None, mode="load", path='../word_embedding/data/FastText_model.bin')
ft_data_loader = FastTextDataLoader('../word_embedding/data/FastText_model.bin')
x, y = ft_data_loader.create_train_data()
x = x[0:250]
y = y[0:250]
x_embedding = []
for text in tqdm(x):
    x_embedding.append(ft_model.get_query_embedding(text))
x_embedding = np.array(x_embedding)
#x_embedding = np.array([ft_model.get_query_embedding(text) for text in tqdm(x)])

# 1. Dimension Reduction
# TODO: Perform Principal Component Analysis (PCA):
#     - Reduce the dimensionality of features using PCA. (you can use the reduced feature afterward or use to the whole embeddings)
#     - Find the Singular Values and use the explained_variance_ratio_ attribute to determine the percentage of variance explained by each principal component.
#     - Draw plots to visualize the results.
dimension_reduction = DimensionReduction()
x = dimension_reduction.pca_reduce_dimension(x_embedding, 60)
dimension_reduction.wandb_plot_explained_variance_by_components(x_embedding, "IMDb-IR", "Phase2")

# TODO: Implement t-SNE (t-Distributed Stochastic Neighbor Embedding):
#     - Create the convert_to_2d_tsne function, which takes a list of embedding vectors as input and reduces the dimensionality to two dimensions using the t-SNE method.
#     - Use the output vectors from this step to draw the diagram.
x_tsne = dimension_reduction.convert_to_2d_tsne(x)
dimension_reduction.wandb_plot_2d_tsne(x, "IMDb-IR", "Phase2")

# 2. Clustering
## K-Means Clustering
# TODO: Implement the K-means clustering algorithm from scratch.
# TODO: Create document clusters using K-Means.
# TODO: Run the algorithm with several different values of k.
# TODO: For each run:
#     - Determine the genre of each cluster based on the number of documents in each cluster.
#     - Draw the resulting clustering using the two-dimensional vectors from the previous section.
#     - Check the implementation and efficiency of the algorithm in clustering similar documents.
# TODO: Draw the silhouette score graph for different values of k and perform silhouette analysis to choose the appropriate k.
# TODO: Plot the purity value for k using the labeled data and report the purity value for the final k. (Use the provided functions in utilities)
cluster_utils = ClusteringUtils()
k_values = []
for i in range(2, 10, 2):
    k_values.append(i)

for k in k_values:
    cluster_utils.visualize_kmeans_clustering_wandb(x, k, "IMDb-IR", "Phase2")

cluster_utils.plot_kmeans_cluster_scores(x, y, k_values, "IMDb-IR", "Phase2")

## Hierarchical Clustering
# TODO: Perform hierarchical clustering with all different linkage methods.
# TODO: Visualize the results.
linkages = ["single", "complete", "average", "ward"]
for linkage in linkages:
    cluster_utils.wandb_plot_hierarchical_clustering_dendrogram(x, "IMDb-IR", linkage, "Phase2")

# 3. Evaluation
# TODO: Using clustering metrics, evaluate how well your clustering method is performing.
cluster_utils.visualize_elbow_method_wcss(x, [2, 4, 6, 8], "IMDb-IR", "Phase2")
cluster_metrics = ClusteringMetrics()
k = 8
_, assignments = cluster_utils.cluster_kmeans(x, k)
print(f"{k}-Means Clustering:  silhouette: {cluster_metrics.silhouette_score(x, assignments)} , purity: {cluster_metrics.purity_score(y, assignments)} , adjusted: {cluster_metrics.adjusted_rand_score(y, assignments)}")
