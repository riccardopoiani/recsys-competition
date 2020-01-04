
from course_lib.Base.Recommender_utils import check_matrix, similarityMatrixTopK
from course_lib.Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender

from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
from sklearn.preprocessing import normalize as normalize_sk

from src.model.Recommender_utils import userSimilarityMatrixTopK


class UserKNNDotCFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNNDotCFRecommender """

    RECOMMENDER_NAME = "UserKNNDotCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, verbose=True):
        super(UserKNNDotCFRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, topK=50, shrink=100, normalize=True, feature_weighting="none"):

        self.topK = topK
        self.shrink = shrink

        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train)
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train)
            self.URM_train = check_matrix(self.URM_train, 'csr')

        denominator = 1 if shrink == 0 else shrink

        self.W_sparse = self.URM_train.dot(self.URM_train.T) * (1/denominator)

        if self.topK >= 0:
            self.W_sparse = userSimilarityMatrixTopK(self.W_sparse, k=self.topK).tocsr()

        if normalize:
            self.W_sparse = normalize_sk(self.W_sparse, norm="l1")

        self.W_sparse = check_matrix(self.W_sparse, format='csr')
