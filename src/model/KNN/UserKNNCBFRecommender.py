from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Base.BaseSimilarityMatrixRecommender import BaseUserSimilarityMatrixRecommender
from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps

from course_lib.Base.Similarity.Compute_Similarity import Compute_Similarity


class UserKNNCBFRecommender(BaseUserSimilarityMatrixRecommender):
    """ UserKNN recommender"""

    RECOMMENDER_NAME = "UserKNNCBFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, UCM_train, verbose=True):
        super(UserKNNCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.UCM_train = UCM_train

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            interactions_feature_weighting="none", field_weights: np.ndarray=None, **similarity_args):

        self.topK = topK
        self.shrink = shrink

        if interactions_feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, interactions_feature_weighting))

        if interactions_feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train)
            self.URM_train = check_matrix(self.URM_train, 'csr')

        elif interactions_feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train)
            self.URM_train = check_matrix(self.URM_train, 'csr')

        # Apply statistical feature weighting
        if feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, feature_weighting))

        if feature_weighting == "BM25":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = okapi_BM_25(self.UCM_train)

        elif feature_weighting == "TF-IDF":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = TF_IDF(self.UCM_train)

        similarity = Compute_Similarity(self.UCM_train.T, shrink=shrink, topK=topK, normalize=normalize,
                                        similarity=similarity, **similarity_args)

        self.W_sparse = similarity.compute_similarity()
        self.W_sparse = check_matrix(self.W_sparse, format='csr')

