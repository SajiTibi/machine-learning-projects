import copy
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_blobs, load_digits
from sklearn.metrics.pairwise import euclidean_distances
import scipy


def circles_example():
    """
	an example function for generating and plotting synthetic data.
	"""

    t = np.arange(0, 2 * np.pi, 0.03)
    length = np.shape(t)
    length = length[0]
    circle1 = np.matrix([np.cos(t) + 0.1 * np.random.randn(length),
                         np.sin(t) + 0.1 * np.random.randn(length)])
    circle2 = np.matrix([2 * np.cos(t) + 0.1 * np.random.randn(length),
                         2 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle3 = np.matrix([3 * np.cos(t) + 0.1 * np.random.randn(length),
                         3 * np.sin(t) + 0.1 * np.random.randn(length)])
    circle4 = np.matrix([4 * np.cos(t) + 0.1 * np.random.randn(length),
                         4 * np.sin(t) + 0.1 * np.random.randn(length)])
    circles = np.concatenate((circle1, circle2, circle3, circle4), axis=1)

    # plt.plot(circles[0, :], circles[1, :], '.k')
    # plt.show()
    return circles


def apml_pic_example(path='APML_pic.pickle'):
    """
	an example function for loading and plotting the APML_pic data.

	:param path: the path to the APML_pic.pickle file
	"""

    with open(path, 'rb') as f:
        apml = pickle.load(f)

    plt.plot(apml[:, 0], apml[:, 1], '.')
    plt.show()


def microarray_exploration(data_path='microarray_data.pickle',
                           genes_path='microarray_genes.pickle',
                           conds_path='microarray_conds.pickle'):
    """
	an example function for loading and plotting the microarray data.
	Specific explanations of the data set are written in comments inside the
	function.

	:param data_path: path to the data file.
	:param genes_path: path to the genes file.
	:param conds_path: path to the conds file.
	"""

    # loading the data:
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    with open(genes_path, 'rb') as f:
        genes = pickle.load(f)
    with open(conds_path, 'rb') as f:
        conds = pickle.load(f)

    # look at a single value of the data matrix:
    i = 128
    j = 63
    print("gene: " + str(genes[i]) + ", cond: " + str(
        conds[0, j]) + ", value: " + str(data[i, j]))

    # make a single bar plot:
    plt.figure()
    plt.bar(np.arange(len(data[i, :])) + 1, data[i, :])
    plt.show()

    # look at two conditions and their correlation:
    plt.figure()
    plt.scatter(data[:, 27], data[:, 29])
    plt.plot([-5, 5], [-5, 5], 'r')
    plt.show()

    # see correlations between conds:
    correlation = np.corrcoef(np.transpose(data))
    plt.figure()
    plt.imshow(correlation, extent=[0, 1, 0, 1])
    plt.colorbar()
    plt.show()

    # look at the entire data set:
    plt.figure()
    plt.imshow(data, extent=[0, 1, 0, 1], cmap="hot", vmin=-3, vmax=3)
    plt.colorbar()
    plt.show()


def euclid(X, Y):
    """
	return the pair-wise euclidean distance between two data matrices.
	:param X: NxD matrix.
	:param Y: MxD matrix.
	:return: NxM euclidean distance matrix.
	"""

    return euclidean_distances(X, Y)


def euclidean_centroid(X):
    """
	return the center of mass of data points of X.
	:param X: a sub-matrix of the NxD data matrix that defines a cluster.
	:return: the centroid of the cluster.
	"""
    return np.mean(X, axis=0)


def kmeans_pp_init(X, k, metric):
    """
	The initialization function of kmeans++, returning k centroids.
	:param X: The data matrix.
	:param k: The number of clusters.
	:param metric: a metric function like specified in the kmeans documentation.
	:return: kxD matrix with rows containing the centroids.
	"""
    N, D = X.shape
    centers = np.zeros((k, X.shape[1]))
    centers[0] = X[np.random.choice(X.shape[0])]
    for i in range(1, k):
        distance_vector = np.empty((i, X.shape[0]))
        for j in range(i):
            distance_vector[j] = metric(X, centers[j].reshape(1, -1)).reshape(1, -1)
        distance_vector = np.min(distance_vector, axis=0)
        distance_vector = distance_vector ** 2
        distance_vector = distance_vector / distance_vector.sum()
        new_ci = np.random.choice(N, p=distance_vector)
        centers[i] = X[new_ci]
    return centers


def kmeans(X, k, iterations=10, metric=euclid, center=euclidean_centroid, init=kmeans_pp_init):
    """
	The K-Means function, clustering the data X into k clusters.
	:param X: A NxD data matrix.
	:param k: The number of desired clusters.
	:param iterations: The number of iterations.
	:param metric: A function that accepts two data matrices and returns their
			pair-wise distance. For a NxD and KxD matrices for instance, return
			a NxK distance matrix.
	:param center: A function that accepts a sub-matrix of X where the rows are
			points in a cluster, and returns the cluster centroid.
	:param init: A function that accepts a data matrix and k, and returns k initial centroids.
	:param stat: A function for calculating the statistics we want to extract about
				the result (for K selection, for example).
	:return: a tuple of (clustering, centroids, statistics)
	clustering - A N-dimensional vector with indices from 0 to k-1, defining the clusters.
	centroids - The kxD centroid matrix.
	"""

    centroids = init(X, k, metric)
    clusters = None
    for i in range(iterations):
        clusters = np.argmin(metric(X, centroids), axis=1)
        old_centroids = copy.deepcopy(centroids)
        for j in range(k):
            count_j_in_cluster = np.count_nonzero(clusters == j)
            matched_xs = np.empty((count_j_in_cluster, X.shape[1]))
            counter = 0
            for m in range(len(X)):
                if clusters[m] == j:
                    matched_xs[counter] = X[m]
                    counter += 1
            centroids[j] = center(matched_xs)
        # check for convergence
        if np.all(centroids - old_centroids) < 0.01:
            return clusters, centroids

    return clusters, centroids


def gaussian_kernel(X, sigma):
    """
	calculate the gaussian kernel similarity of the given distance matrix.
	:param X: A NxN distance matrix.
	:param sigma: The width of the Gaussian in the heat kernel.
	:return: NxN similarity matrix.
	"""

    return scipy.exp(-X ** 2 / (2 * sigma ** 2))


def mnn(X, m):
    """
	calculate the m nearest neighbors similarity of the given distance matrix.
	:param X: A NxN distance matrix.
	:param m: The number of nearest neighbors.
	:return: NxN similarity matrix.
	"""
    res = np.zeros(X.shape)
    for i in range(X.shape[0]):
        res[i, np.argsort(X[i, :])[1:m + 1]] = 1

    return (res + res.T > 0).astype(np.float64)


def spectral(X, k, similarity_param, similarity=mnn):
    """
	Cluster the data into k clusters using the spectral clustering algorithm.
	:param X: A NxD data matrix.
	:param k: The number of desired clusters.
	:param similarity_param: m for mnn, sigma for the Gaussian kernel.
	:param similarity: The similarity transformation of the data.
	:return: clustering, as in the kmeans implementation.
	"""
    N = X.shape[0]
    S = euclid(X, X)
    W = similarity(S, similarity_param)
    D = np.zeros(W.shape)
    for i in range(N):
        D[i, i] = np.sum(W[i])
    I = np.eye(N)
    new_D = D ** (-1 / 2)
    new_D[new_D == np.inf] = 0
    L = I - new_D.dot(W).dot(new_D)
    eigenvalues, eigenvectors = scipy.linalg.eigh(L, eigvals=(0, k - 1))
    clustering, _ = kmeans(eigenvectors, k, )
    return clustering


def elbow_method(X, K, iters, data_name):
    loss = []
    for k in range(2, K):
        clusters, centeroids = kmeans(X, k, iters)
        current_loss = 0
        for j in range(k):
            matched_xs = X[clusters == j]
            current_loss += np.sum(euclid(matched_xs, centeroids[j].reshape(1, -1)))
        loss.append(current_loss)
    plt.xlabel("k")
    plt.ylabel("Cost")
    plt.title(data_name + " elbow")
    plt.plot(np.arange(2, K), loss)
    plt.show()


def silhouette(X, K, iters, data_name):
    score = []
    for k in range(2, K):
        clusters, centroids = kmeans(X, k, iters)
        A = np.zeros((X.shape[0]))
        B = np.zeros((X.shape[0]))
        for j in range(k):
            matched_xs = X[clusters == j]
            distances = euclid(matched_xs, matched_xs)
            distances = np.sum(distances, axis=0)
            distances = distances / ((matched_xs.shape[0]) - 1)
            A[clusters == j] = distances
        for j in range(k):
            matched_xs = X[clusters == j]
            current_cluster_b = np.zeros((k, matched_xs.shape[0]))
            for c in range(k):
                if c != j:
                    extraterrestrial_xs = X[clusters == c]
                    current_b = np.sum(euclid(matched_xs, extraterrestrial_xs), axis=1)
                    current_b = current_b / extraterrestrial_xs.shape[0]
                    current_cluster_b[c] = current_b
                if c == j:
                    current_cluster_b[c] = np.inf
            B[clusters == j] = np.min(current_cluster_b, axis=0)
        iter_score = np.sum((B - A) / np.maximum(A, B)) / X.shape[0]
        score.append(iter_score)
    plt.xlabel("k")
    plt.ylabel("Cost")
    plt.title(data_name + " Silhouette")
    plt.plot(np.arange(2, K), score)
    plt.show()


def eigen_gap(X, similarity_param, similarity=mnn):
    max_clusters = 15  # defined in exercise
    N = X.shape[0]
    S = euclid(X, X)
    W = similarity(S, similarity_param)

    D = np.zeros(W.shape)
    for i in range(N):
        D[i, i] = np.sum(W[i])
    I = np.eye(N)
    new_D = D ** (-1 / 2)

    new_D[new_D == np.inf] = 0
    L = I - new_D.dot(W).dot(new_D)
    eigenvalues, eigenvectors = scipy.linalg.eigh(L, eigvals=(0, max_clusters - 1))
    plt.plot(eigenvalues)
    plt.show()


def kmeans_clustering_demonstration(X, k):
    clusters = kmeans(X, k, iterations=10)[0]
    colormap = plt.cm.get_cmap('rainbow', (k + 1))
    for i in range(k):
        plt.plot(X[clusters == i, 0], X[clusters == i, 1], '.', color=colormap(i))
    plt.show()


def kmeans_clustering_demonstration_micro(X, k):
    clusters= kmeans(X, k, iterations=10)[0]
    for i in range(k):
        current_clutser = X[clusters == i]
        cluster_title = "biological data kmeans++, cluster#: " + str(i + 1) + " size: " + str(current_clutser.shape[0])
        plt.figure()
        plt.title(cluster_title)
        plt.imshow(current_clutser, extent=[0, 1, 0, 1], vmin=-3, vmax=3, cmap="hot")
    plt.show()


def spectral_clustering_demonstration(X, k, similiarity_param):
    clusters = spectral(X, k, similiarity_param)
    colormap = plt.cm.get_cmap('rainbow', (k + 1))
    for i in range(k):
        plt.plot(X[clusters == i, 0], X[clusters == i, 1], '.', color=colormap(i))
    plt.show()

def kmeans_spectral_clustering_demonstration_micro(X, k):
    clusters = spectral(X, k, 15)
    for i in range(k):
        current_cluster = X[clusters == i]
        cluster_title = "biological data spectral clustering, cluster#: " + str(i + 1) + " size: " + str(
            current_cluster.shape[0])
        plt.figure()
        plt.title(cluster_title)
        plt.imshow(current_cluster, extent=[0, 1, 0, 1], vmin=-3, vmax=3, cmap="hot")
    plt.show()


def similiarity_plot():
    data, _ = make_blobs(n_samples=1000, n_features=2, centers=3)
    shuffled_data = np.random.permutation(data)
    similarity_matrix = mnn(euclid(shuffled_data, shuffled_data), 10)  # why 10? explained in pdf
    plt.imshow(similarity_matrix)
    plt.show()
    clustered = spectral(data, 3, 10, mnn)  # params explained in pdf
    sorted_clusters = data[np.argsort(clustered)]
    similarity_matrix = mnn(euclid(sorted_clusters, sorted_clusters), 10)
    plt.imshow(similarity_matrix)
    plt.show()


def tsne_compare():
    digits = load_digits()
    digits_labels = digits.target
    t_SNE_embedding = TSNE().fit_transform(digits.data, digits_labels)
    PCA_embedding = PCA().fit_transform(digits.data, digits_labels)

    plt.subplot(1, 2, 1)
    plt.title("t-SNE embedding")
    plt.scatter(t_SNE_embedding[:, 0], t_SNE_embedding[:, 1], c=digits_labels, cmap="hsv")

    plt.subplot(1, 2, 2)
    plt.title("PCA embedding")
    plt.scatter(PCA_embedding[:, 0], PCA_embedding[:, 1], c=digits_labels, cmap="hsv")
    plt.show()


if __name__ == '__main__':
    pass

    # ignore code below, to ran or test please use whats in excerise and with parameters found in pdf
    # kmeans++
    # Synthetic data
    #max_k = 10
    #iters = 10
    # data, _ = make_blobs(n_samples=1000, n_features=2, centers=3)
    # X = np.matrix(data)
    # elbow_method(X, max_k,iters,"Synthetic Data")
    # silhouette(X, max_k,iters,"Synthetic Data")
    # kmeans_clustering_demonstration(X, 5)  # using k =5 since that we concluded from cost graphs
    # eigen_gap(X,0.6,gaussian_kernel)
    # circles
    # X = circles_example().T
    # elbow_method(X, max_k, iters, "Circle Data")
    # silhouette(X, max_k, iters, "Circle Data")
    # kmeans_clustering_demonstration(X, 4)  # using k =4 since that we concluded from cost graphs
    #
   # APML
   #  with open('APML_pic.pickle', 'rb') as f:
   #      data = pickle.load(f)
   #  X = np.matrix(data)
   #  elbow_method(X, max_k, iters, "APML PICTURE")
   #  silhouette(X, max_k, iters, "APML PICTURE")
   #  kmeans_clustering_demonstration(X, 9)  # using k =9 since that we concluded from cost graphs

    # data, _ = make_blobs(n_samples=1000, n_features=2, centers=3)
    # X = np.matrix(data)
    # elbow_method(X, max_k,iters,"Synthetic Data")
    # silhouette(X, max_k,iters,"Synthetic Data")
    # spectral_clustering_demonstration(X, 5)  # using k =5 since that we concluded from cost graphs

    # # APML
    # with open('APML_pic.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # X = np.matrix(data)
    # elbow_method(X, max_k, iters, "Circle Data")
    # silhouette(X,max_k,iters,"Cricle data")
    # spectral_clustering_demonstration(X, 9,16)  # using k =9 since that we concluded from cost graphs

    # circles
    # X = circles_example().T
    # eigen_gap(X,0.1,gaussian_kernel)
    # elbow_method(X, max_k, iters, "Circle Data")
    # silhouette(X, max_k, iters, "Circle Data")
    # spectral_clustering_demonstration(X, 4,0.1)  # using k =4 since that we concluded from cost graph
    # similiarity_plot()

    # with open('microarray_data.pickle', 'rb') as f:
    #     data = pickle.load(f)
    # X = np.matrix(data)
    # kmeans_spectral_clustering_demonstration_micro(X,6)
    # elbow_method(X, max_k, iters, "Micro Data")
    # eigen_gap(X,14,mnn)
    # silhouette(X,max_k,iters,"Micro data")
    # kmeans_clustering_demonstration(X, 9)  # using k =8 since that we concluded from cost graphs
    # tsne_compare()
