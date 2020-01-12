import scipy.sparse as sps
from sklearn.utils.extmath import randomized_svd

from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from src.model.Recommender_utils import apply_feature_weighting
from src.utils.general_utility_functions import get_split_seed


class NewPureSVDRecommender(BaseMatrixFactorizationRecommender):
    """ New PureSVDRecommender"""

    RECOMMENDER_NAME = "NewPureSVDRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, verbose=True):
        super(NewPureSVDRecommender, self).__init__(URM_train, verbose=verbose)

    def fit(self, num_factors=100, n_oversamples=10, n_iter=4, feature_weighting="none", random_seed=get_split_seed()):
        self._print("Computing SVD decomposition...")

        self.URM_train = apply_feature_weighting(self.URM_train, feature_weighting)

        U, Sigma, VT = randomized_svd(self.URM_train,
                                      n_oversamples=n_oversamples,
                                      n_iter=n_iter,
                                      n_components=num_factors,
                                      random_state=random_seed)

        s_Vt = sps.diags(Sigma) * VT

        self.USER_factors = U
        self.ITEM_factors = s_Vt.T

        self._print("Computing SVD decomposition... Done!")
