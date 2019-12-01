from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import implicit
import numpy as np

from course_lib.Base.Recommender_utils import check_matrix


class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ Implicit ALS Recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    AVAILABLE_CONFIDENCE_SCALING = ["linear", "log"]

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

    def fit(self, epochs=300, num_factors=50, regularization=0.01, confidence_scaling="linear", alpha=1.0, epsilon=1.0):
        if confidence_scaling not in self.AVAILABLE_CONFIDENCE_SCALING:
            raise ValueError(
                "Value for 'confidence_scaling' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.AVAILABLE_CONFIDENCE_SCALING, confidence_scaling))

        self.alpha = alpha
        self.epsilon = epsilon

        self._build_confidence_matrix(confidence_scaling)

        self.model = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=regularization,
                                                          iterations=epochs, calculate_training_loss=True)
        self.model.fit(self.C_iu)
        self.USER_factors = self.model.user_factors
        self.ITEM_factors = self.model.item_factors

    def _build_confidence_matrix(self, confidence_scaling):
        if confidence_scaling == 'linear':
            self.C = self._linear_scaling_confidence()
        else:
            self.C = self._log_scaling_confidence()

        self.C_iu = check_matrix(self.C.transpose().copy(), format="csr", dtype=np.float32)

    def _linear_scaling_confidence(self):
        C = check_matrix(self.URM_train, format="csr", dtype=np.float32)
        C.data = 1.0 + self.alpha * C.data
        return C

    def _log_scaling_confidence(self):
        C = check_matrix(self.URM_train, format="csr", dtype=np.float32)
        C.data = 1.0 + self.alpha * np.log(1.0 + C.data / self.epsilon)
        return C
