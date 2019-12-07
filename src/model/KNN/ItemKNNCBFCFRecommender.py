
import scipy.sparse as sps

from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender


class ItemKNNCBFCFRecommender(ItemKNNCBFRecommender):
    """ Item KNN CBF CF recommender"""

    RECOMMENDER_NAME = "ItemKNNCBFCFRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, ICM_train, verbose=True):
        super(ItemKNNCBFRecommender, self).__init__(URM_train, verbose=verbose)

        self.ICM_train = sps.hstack([ICM_train, URM_train.T], format="csr")

