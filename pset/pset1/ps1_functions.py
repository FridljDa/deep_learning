"""
Deep Learning Theory and Applications, Problem Set 1
"""
# helpful libraries
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter

def problem2_evaluate_function_on_random_noise(N, sigma):
    """Sample N points uniformly from the interval [-1,3],
    and output the function y = x^2 - 3x + 1 with random noise added to the outputs
    Hint: You can sort x before evaluate the function. This could help plot
    smooth polynomial lines later on

    Parameters
    ----------
    N : int
        The number of points
    sigma : float
        The standard deviation of noise to add to the randomly generated points.

    Returns
    -------
    x, y (list, list)
        x, the randomly generated points
        y, the function evaluated at these points, with added noise
    """

    # Sample N points uniformly from the interval [-1,3],
    x = np.random.uniform(-1, 3, N)
    y = x * x - 3 * x + 1 + np.random.normal(0, sigma, N)
    return x, y



def problem2_fit_polynomial(x, y, degree, regularization = 0):
    """Returns optimal coefficients for a polynomial of the given degree
    to fit the data, using the Moore-Penrose Pseudoinverse (specified in the assignment)
    Note: this function only needs to function for degrees 1,2, and 9 --
    but you are welcome build something that works for any degree.
    By incorporating the value of the regularization parameter, this function should work
    for both 2.2 and 2.3

    Parameters
    ----------
    x : list of floats
        The input x values
    y : list of floats
        The input y values
    degree : int
        The degree of the polynomial to fit
    regularization : float
        The parameter lambda which specifies the degree of regularization to apply. Default 0.

    Returns
    -------
    coeffs : list of floats
        The coefficients of the polynomial.
    """
    degree = degree + 1
    #matrix with degree columns and ith row is x[i]**n
    X_matrix = np.array([X_i ** n for X_i in x for n in range(0, degree)]).reshape(len(x),degree)

    #pseudo inverse of X_matrix

    X_matrix_pinv = np.linalg.inv(X_matrix.T @ X_matrix + regularization * np.identity(degree)) @ X_matrix.T


    coeffs = np.dot(X_matrix_pinv, y)

    return coeffs




def problem3_knn_classifier(train_data, train_labels, test_data, k):
    """A kth Nearest Neighbor classified. Accepts points and training labels,
    and returns predicted labels for each point in the dataset.

    Parameters
    ----------
    train_data : ndarray
        The training points, in an n x d array, where n is the number of points and d is the dimension.
    train_labels : list of classes
        The training labels. They should correspond directly to the points in the training data array.
    test_data : ndarray
        The unlabelled data, to be labelled by the classifier
    k : positive int
        The number of nearest neighbors to consult.

    Returns
    -------
    predicted_labels : list
        The labels outputted by the classifier for each of the test datapoints.
    """
    neigh = KNeighborsClassifier(n_neighbors=k)
    neigh.fit(train_data, train_labels)

    distances, indices = neigh.kneighbors(test_data)
    class_candidates = train_labels[indices]

    #number of rows in class_candidates
    num_rows = class_candidates.shape[0]
    #vector of size num_rows to store the most common class
    most_common_class = np.zeros(num_rows)

    # for eac row in class_candidates find the most common class
    for i in range(num_rows):
        most_common_class[i], occurance = Counter(class_candidates[i,:]).most_common(1)[0]

    return most_common_class
