import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import euclidean_distances
import scipy
from keras.datasets import mnist
from sklearn.datasets import load_digits


def digits_example():
    '''
	Example code to show you how to load the MNIST data and plot it.
	'''

    # load the MNIST data:
    digits = datasets.load_digits()
    data = digits.data / 255.
    labels = digits.target

    # plot examples:
    plt.gray()
    for i in range(10):
        plt.subplot(2, 5, i + 1)
        plt.axis('off')
        plt.imshow(np.reshape(data[i, :], (8, 8)))
        plt.title("Digit " + str(labels[i]))
    plt.show()


def swiss_roll_example():
    '''
	Example code to show you how to load the swiss roll data and plot it.
	'''

    # load the dataset:
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)

    # plot the data:
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
    plt.show()


def faces_example(path):
    '''
	Example code to show you how to load the faces data.
	'''

    with open(path, 'rb') as f:
        X = pickle.load(f)

    num_images, num_pixels = np.shape(X)
    d = int(num_pixels ** 0.5)
    print("The number of images in the data set is " + str(num_images))
    print("The image size is " + str(d) + " by " + str(d))

    # plot some examples of faces:
    plt.gray()
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.imshow(np.reshape(X[i, :], (d, d)))
    plt.show()


def plot_with_images(X, images, title, image_num=25):
    '''
	A plot function for viewing images in their embedded locations. The
	function receives the embedding (X) and the original images (images) and
	plots the images along with the embeddings.

	:param X: Nxd embedding matrix (after dimensionality reduction).
	:param images: NxD original data matrix of images.
	:param title: The title of the plot.
	:param num_to_plot: Number of images to plot along with the scatter plot.
	:return: the figure object.
	'''

    n, pixels = np.shape(images)
    img_size = int(pixels ** 0.5)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # get the size of the embedded images for plotting:
    x_size = (max(X[:, 0]) - min(X[:, 0])) * 0.08
    y_size = (max(X[:, 1]) - min(X[:, 1])) * 0.08

    # draw random images and plot them in their relevant place:
    for i in range(image_num):
        img_num = np.random.choice(n)
        x0, y0 = X[img_num, 0] - x_size / 2., X[img_num, 1] - y_size / 2.
        x1, y1 = X[img_num, 0] + x_size / 2., X[img_num, 1] + y_size / 2.
        img = images[img_num, :].reshape(img_size, img_size)
        ax.imshow(img, aspect='auto', cmap=plt.cm.gray, zorder=100000,
                  extent=(x0, x1, y0, y1))

    # draw the scatter plot of the embedded data points:
    ax.scatter(X[:, 0], X[:, 1], marker='.', alpha=0.7)

    return fig


def euclid(X, Y):
    """
	return the pair-wise euclidean distance between two data matrices.
	:param X: NxD matrix.
	:param Y: MxD matrix.
	:return: NxM euclidean distance matrix.
	"""
    return euclidean_distances(X, Y)


def MDS(X, d):
    '''
	Given a NxN pairwise distance matrix and the number of desired dimensions,
	return the dimensionally reduced data points matrix after using MDS.

	:param X: NxN distance matrix.
	:param d: the dimension.
	:return: Nxd reduced data point matrix.
	'''
    n, _ = X.shape
    In = np.eye(n)
    H =In - 1 / n * (In *In.T)
    S = -1 / 2 * (np.matmul((np.matmul(H, X)), H))
    eigenvalues, eigenvectors = np.linalg.eigh(S)
    values_idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[values_idxs]
    eigenvectors = eigenvectors[:, values_idxs[:d]]
    reduced_mat = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues[:d])))
    return eigenvalues, reduced_mat


# deprecated code, needs modifying and would work with less sortings
# def MDS(X, d):
# 	'''
# 	Given a NxN pairwise distance matrix and the number of desired dimensions,
# 	return the dimensionally reduced data points matrix after using MDS.
#
# 	:param X: NxN distance matrix.
# 	:param d: the dimension.
# 	:return: Nxd reduced data point matrix.
# 	'''
# 	n, _ = X.shape
# 	H = np.eye(n) - 1 / n * (np.ones(n) * np.ones(n).T)
# 	S = -1 / 2 * (np.matmul((np.matmul(H, X)), H))
# 	eigenvalues, eigenvectors = scipy.linalg.eigh(S,eigvals=(0,n-1))
# 	eigenvectors = eigenvectors[:,d]
#
# 	reduced_mat = np.dot(eigenvectors, np.diag(np.sqrt(eigenvalues[:d])))
# 	return eigenvalues, reduced_mat

def LLE(X, d, k):
    '''
	Given a NxD data matrix, return the dimensionally reduced data matrix after
	using the LLE algorithm.

	:param X: NxD data matrix.
	:param d: the dimension.
	:param k: the number of neighbors for the weight extraction.
	:return: Nxd reduced data matrix.
	'''
    n, D = np.shape(X)
    dists = euclid(X, X)
    for i in range(n):
        dists[i, i] = np.inf
    knn_indices = np.zeros((n, k), dtype=np.int)
    for i in range(n):
        knn_indices[i, :] = np.argpartition(dists[i], k)[:k]
    W = np.zeros((n, n))
    for i in range(n):
        xi = X[i, :]
        xjs = X[knn_indices[i], :]
        zi = xi - xjs
        G_inverse = np.linalg.pinv(np.dot(zi, zi.T))
        ones_vector = np.ones(k)
        W[i, knn_indices[i]] = np.dot(G_inverse, ones_vector)
        W[i] = W[i] / np.sum(W[i])
    In = np.eye(n)
    MTM = np.dot((In - W).T, (In - W))
    # eigvals parameter would return 1 to d eigenvalues and eigenvectors that correspond to
    # lowest first d eigenvalues after first eigenvalue which is
    eigenvalues, eigenvectors = scipy.linalg.eigh(MTM, eigvals=(1, d))
    return eigenvectors


def gaussian_kernel(X, sigma):
    """
    ex4
	calculate the gaussian kernel similarity of the given distance matrix.
	:param X: A NxN distance matrix.
	:param sigma: The width of the Gaussian in the heat kernel.
	:return: NxN similarity matrix.
	"""
    return scipy.exp(-X ** 2 / (2 * sigma ** 2))


def DiffusionMap(X, d, sigma, t):
    '''
	Given a NxD data matrix, return the dimensionally reduced data matrix after
	using the Diffusion Map algorithm. The k parameter allows restricting the
	kernel matrix to only the k nearest neighbor of each data point.

	:param X: NxD data matrix.
	:param d: the dimension.
	:param sigma: the sigma of the gaussian for the kernel matrix transformation.
	:param t: the scale of the diffusion (amount of time steps).
	:return: Nxd reduced data matrix.
	'''
    n, D = X.shape
    dists = euclid(X, X)
    K = gaussian_kernel(dists, sigma)
    K_rows = np.sum(K, axis=1)
    D = np.diag(K_rows)
    D_i = np.linalg.pinv(D)
    A = np.dot(D_i, K)
    # d eigenvectors corresponds to high d  eigenvalues
    right_eigenvalues, right_eigenvectors = scipy.linalg.eigh(A, eigvals=(n - d, n - 1))
    lambdas_eigenvals = np.diag(right_eigenvalues ** t)
    vectors_multiplied = np.dot(right_eigenvectors, lambdas_eigenvals)
    return vectors_multiplied


def scree_plot():
    N = 10
    random_data = np.random.randn(N, N)
    added_noise = np.random.rand(N, N)
    Q, R = scipy.linalg.qr(random_data)
    q_composed = np.dot(random_data, Q)
    sigmas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100]
    # plt.suptitle("Scree plot of eigenvalues of MDS as function of noise")
    for i in range(len(sigmas)):
        plt.subplot(5, 2, 1 + i)
        plt.title("sigma = " + str(sigmas[i]))
        X = q_composed + sigmas[i] * added_noise
        eigenvalues, _ = MDS(euclid(X, X) ** 2, 2)
        eigenvalues = np.abs(eigenvalues)
        idx = np.argsort(eigenvalues)[::-1]
        plt.plot(eigenvalues[idx])
    plt.show()


def mnist_compare():
    mnist_data = load_digits()
    mnist_digits = mnist_data.data
    mnist_labels = mnist_data.target

    # MDS
    _, mds = MDS(euclid(mnist_digits, mnist_digits) ** 2, 2)
    plt.scatter(mds[:, 0], mds[:, 1], c=mnist_labels, cmap='hsv')
    plt.title("MDS on MNIST")
    plt.show()

    # LLE
    k = 15  # see parameter tweaking why choose this k
    lle = LLE(mnist_digits, 2, k)
    plt.scatter(lle[:, 0], lle[:, 1], c=mnist_labels, cmap='hsv')
    plt.title("LLE on MNIST")
    plt.show()

    # DM
    dm = DiffusionMap(mnist_digits, 2, 12, 2)
    plt.scatter(dm[:, 0], dm[:, 1], c=mnist_labels, cmap='hsv')
    plt.title("DM on MNIST")
    plt.show()


def swiss_compare():
    X, color = datasets.samples_generator.make_swiss_roll(n_samples=2000)
    # MDS
    _, mds = MDS(euclid(X, X) ** 2, 2)
    plt.scatter(mds[:, 0], mds[:, 1], c=color, cmap='hsv')
    plt.title("MDS on swiss roll")
    plt.show()

    # LLE
    k = 15
    lle = LLE(X, 2, k)
    plt.scatter(lle[:, 0], lle[:, 1], c=color, cmap='hsv')
    plt.title("LLE on swiss roll")
    plt.show()

    # DM
    dm = DiffusionMap(X, 2, 2, 2)
    plt.scatter(dm[:, 0], dm[:, 1], c=color, cmap='hsv')
    plt.title("DM on swiss roll")
    plt.show()


def compare_faces():
    with open('faces.pickle', 'rb') as f:
        faces = pickle.load(f)
    # MDS
    image_num = int(len(faces) * 0.1)  # only 10% of images to show
    _, mds = MDS(euclid(faces, faces) ** 2, 2)
    plot_with_images(mds, faces, "MDS on faces", image_num)
    plt.show()

    # LLE
    lle = LLE(faces, 2, 14)  # see parameter tweaking
    plot_with_images(lle, faces, "LLE on faces", image_num)
    plt.show()

    # DM
    dm = DiffusionMap(faces, 2, 15, 4)
    plot_with_images(dm, faces, "DM on faces", image_num)
    plt.show()


def LLE_parameter_tweaking():
    with open('faces.pickle', 'rb') as f:
        faces = pickle.load(f)
    image_num = int(len(faces) * 0.1)  # only 10% of images to show
    LLE_paras = [3, 5, 14, 25, 50]
    for parameter in LLE_paras:
        lle = LLE(faces, 2, parameter)
        plot_with_images(lle, faces, "LLE on faces using k=" + str(parameter), image_num)
        plt.show()


if __name__ == '__main__':
    # scree_plot()
    # mnist_compare()
    # swiss_compare()
    # compare_faces()
    # LLE_parameter_tweaking()
    pass
