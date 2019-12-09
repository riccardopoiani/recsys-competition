import scipy.sparse as sps
import numpy as np

from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class ItemKNNCBFCFRecommender(ItemKNNCBFRecommender):
    """ Item KNN CBF CF recommender"""

    RECOMMENDER_NAME = "ItemKNNCBFCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(ItemKNNCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.ICM_train = sps.hstack([ICM_train, URM_train.T], format="csr")

    def fit(self, topK=50, shrink=100, similarity='cosine', normalize=True, feature_weighting="none",
            interactions_feature_weighting="none", **similarity_args):

        if interactions_feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, interactions_feature_weighting))

        if interactions_feature_weighting == "BM25":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = okapi_BM_25(self.URM_train)

        elif interactions_feature_weighting == "TF-IDF":
            self.URM_train = self.URM_train.astype(np.float32)
            self.URM_train = TF_IDF(self.URM_train)

        super().fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize,
                    feature_weighting=feature_weighting, **similarity_args)
