from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
import scipy.sparse as sps


class UserKNNCBFCFRecommender(UserKNNCBFRecommender):
    """ User KNN CBF CF recommender"""

    RECOMMENDER_NAME = "UserKNNCBFCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, UCM_train, verbose=True):
        super(UserKNNCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.UCM_train = sps.hstack([UCM_train, URM_train], format="csr")

