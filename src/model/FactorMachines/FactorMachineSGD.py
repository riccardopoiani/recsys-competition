from abc import ABCMeta, abstractmethod
import numpy as np
from sklearn import metrics
from scipy.sparse import coo_matrix


def preprocess_URM_for_SGD(URM_train, proportion_of_negative_samples=1):
    """
    Format an URM in the way that is needed for the FM model.
    - We have #num_ratings row
    - The last column with all the ratings (for implicit dataset it just a col full of 1
    - In each row there are 3 interactions: 1 for the user, 1 for the item, and 1 for the rating
    - Moreover #num_ratings * proportion_of_negative_samples are inserted, to take into account
    also for this behavior

    Note: this method works only for implicit dataset

    :param URM_train: URM train to be preprocessed
    :return: csr_matrix containing the URM_train preprocessed in the described way
    """
    new_train = URM_train.copy().tocoo()
    fm_matrix = coo_matrix((URM_train.data.size, URM_train.shape[0] + URM_train.shape[1] + 1), dtype=np.int8)

    # Index offset
    item_offset = URM_train.shape[0]

    # Last col
    last_col = URM_train.shape[0] + URM_train.shape[1]

    # Set up initial vectors
    row_v = np.zeros(new_train.data.size * 3)  # Row should have (i,i,i) repeated for all the size
    col_v = np.zeros(new_train.data.size * 3)  # This is the "harder" to set
    data_v = np.ones(new_train.data.size * 3)  # Already ok, nothing to be added

    # Setting row vector
    for i in range(0, new_train.data.size):
        row_v[3 * i] = i
        row_v[(3 * i) + 1] = i
        row_v[(3 * i) + 2] = i

    # Setting col vector
    for i in range(0, new_train.data.size):
        # Retrieving information
        user = new_train.row[i]
        item = new_train.col[i]

        # Fixing col indices to be added to the new matrix
        user_index = user
        item_index = item + item_offset

        col_v[3 * i] = user_index
        col_v[(3 * i) + 1] = item_index
        col_v[(3 * i) + 2] = last_col

    # Setting new information
    fm_matrix.row = row_v
    fm_matrix.col = col_v
    fm_matrix.data = data_v

    return fm_matrix.tocsr()


class SGD:
    __metaclass__ = ABCMeta

    def __init__(self, X, y, l2_reg_w0=0.01, l2_reg_w=0.01, l2_reg_V=0.01, learn_rate=0.01):
        self.X, self.y = X, y
        self.n, self.p = X.shape  # number of samples and their dimension

        # parameter settings
        self.k = int(np.sqrt(self.p))  # for the low-rank assumption of pairwise interaction
        self.l2_reg_w0 = l2_reg_w0
        self.l2_reg_w = l2_reg_w
        self.l2_reg_V = l2_reg_V
        self.learn_rate = learn_rate

        # initialize the model parameters
        # we have the weight vector and the pairwise interaction matrix
        self.w0 = 0.
        self.w = np.zeros(self.p)
        self.V = np.random.random((self.p, self.k))

    def predict(self, x):
        """
        Return a predicted value for an input vector x.
        """

        # efficient vectorized implementation of the pairwise interaction term
        interaction = float(np.sum(np.dot(self.V.T, np.array([x]).T) ** 2 -
                                   np.dot(self.V.T ** 2, np.array([x]).T ** 2)) / 2.)

        # Add the pairwise interaction to the linear weight
        return self.w0 + np.inner(self.w, x) + interaction

    def fit(self):
        """
        Learn the model parameters with SGD. Iterate until convergence.
        """

        prev = float('inf')
        current = 0.
        eps = 1e-3

        history = []
        while abs(prev - current) > eps:
            prev = current
            for x, y in zip(self.X, self.y):  # for each (x, y) training sample
                current = self.update(x, y)
            history.append(current)
        return history

    def update(self, x, y):
        """
        Update the model parameters based on the given vector-value pair.
        """

        # common part of the gradient
        grad_base = self._loss_derivative(x, y)

        grad_w0 = grad_base * 1.
        self.w0 = self.w0 - self.learn_rate * (grad_w0 + 2. * self.l2_reg_w0 * self.w0)

        # Update every paramether
        for i in range(self.p):
            if x[i] == 0.: continue
            grad_w = grad_base * x[i]
            self.w[i] = self.w[i] - self.learn_rate * (grad_w + 2. * self.l2_reg_w * self.w[i])

            for f in range(self.k):
                grad_V = grad_base * x[i] * (sum(x * self.V[:, f]) - x[i] * self.V[i, f])
                self.V[i, f] = self.V[i, f] - \
                               self.learn_rate * (grad_V + 2. * self.l2_reg_V * self.V[i, f])

        return self._evaluate()

    @abstractmethod
    def _loss_derivative(self, x, y):
        # for grad_base
        pass

    @abstractmethod
    def _evaluate(self):
        # RMSE for regression
        # AUC for classification
        pass


class Regression(SGD):
    def _loss_derivative(self, x, y):
        return 2. * (self.predict(x) - y)

    def _evaluate(self):
        # Efficient vectorized RMSE computation
        linear = np.dot(self.X, np.array([self.w]).T).T
        interaction = np.array([np.sum(np.dot(self.X, self.V) ** 2 - np.dot(self.X ** 2, self.V ** 2), axis=1) / 2.])
        y_pred = self.w0 + linear + interaction
        y_pred = y_pred.reshape((self.n))

        return metrics.mean_squared_error(self.y, y_pred) ** 0.5


class Classification(SGD):
    def _loss_derivative(self, x, y):
        return (1. / (1. + np.e ** (- self.predict(x) * y)) - 1) * y

    def _evaluate(self):
        linear = np.dot(self.X, np.array([self.w]).T).T
        interaction = np.array([np.sum(np.dot(self.X, self.V) ** 2 - np.dot(self.X ** 2, self.V ** 2), axis=1) / 2.])
        y_pred = self.w0 + linear + interaction
        y_pred = y_pred.reshape((self.n))

        fpr, tpr, thresholds = metrics.roc_curve(self.y, y_pred, pos_label=1)
        return metrics.auc(fpr, tpr)
