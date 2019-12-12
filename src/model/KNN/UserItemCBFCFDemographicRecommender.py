from course_lib.Base.BaseRecommender import BaseRecommender
from course_lib.Base.DataIO import DataIO
from course_lib.Base.Recommender_utils import check_matrix
from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
import numpy as np
import scipy.sparse as sps

from course_lib.Base.Similarity.Compute_Similarity import Compute_Similarity


class UserItemCBFCFDemographicRecommender(BaseRecommender):
    """ UserItem KNN CBF & CF & Demographic Recommender"""

    RECOMMENDER_NAME = "UserItemCBFCFDemographicRecommender"

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    def __init__(self, URM_train, UCM_train, ICM_train, verbose=True):
        super(UserItemCBFCFDemographicRecommender, self).__init__(URM_train, verbose=verbose)

        self._URM_train_format_checked = False
        self._user_W_sparse_format_checked = False
        self._item_W_sparse_format_checked = False

        self.UCM_train = sps.hstack([UCM_train, URM_train], format="csr")
        self.ICM_train = sps.hstack([ICM_train, URM_train.T], format="csr")

    def fit(self, user_topK=50, user_shrink=100, user_similarity_type='cosine', user_normalize=True,
            user_feature_weighting="none", user_asymmetric_alpha=0.5,
            item_topK=50, item_shrink=100, item_similarity_type='cosine', item_normalize=True,
            item_feature_weighting="none", item_asymmetric_alpha=0.5,
            interactions_feature_weighting="none"):

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

        # User Similarity Computation
        self.user_topK = user_topK
        self.user_shrink = user_shrink

        if user_feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, user_feature_weighting))

        if user_feature_weighting == "BM25":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = okapi_BM_25(self.UCM_train)

        elif user_feature_weighting == "TF-IDF":
            self.UCM_train = self.UCM_train.astype(np.float32)
            self.UCM_train = TF_IDF(self.UCM_train)

        kwargs = {"asymmetric_alpha": user_asymmetric_alpha}
        user_similarity_compute = Compute_Similarity(self.UCM_train.T, shrink=user_shrink, topK=user_topK,
                                                     normalize=user_normalize,
                                                     similarity=user_similarity_type, **kwargs)

        self.user_W_sparse = user_similarity_compute.compute_similarity()
        self.user_W_sparse = check_matrix(self.user_W_sparse, format='csr')

        # Item Similarity Computation
        self.item_topK = item_topK
        self.item_shrink = item_shrink

        if item_feature_weighting not in self.FEATURE_WEIGHTING_VALUES:
            raise ValueError(
                "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                    self.FEATURE_WEIGHTING_VALUES, item_feature_weighting))

        if item_feature_weighting == "BM25":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = okapi_BM_25(self.ICM_train)

        elif item_feature_weighting == "TF-IDF":
            self.ICM_train = self.ICM_train.astype(np.float32)
            self.ICM_train = TF_IDF(self.ICM_train)

        kwargs = {"asymmetric_alpha": item_asymmetric_alpha}
        item_similarity_compute = Compute_Similarity(self.ICM_train.T, shrink=item_shrink, topK=item_topK,
                                                     normalize=item_normalize,
                                                     similarity=item_similarity_type, **kwargs)

        self.item_W_sparse = item_similarity_compute.compute_similarity()
        self.item_W_sparse = check_matrix(self.item_W_sparse, format='csr')

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        """
        URM_train and W_sparse must have the same format, CSR
        :param user_id_array:
        :param items_to_compute: not implemented!!
        :return:
        """

        self._check_format()

        user_weights_array = self.user_W_sparse[user_id_array]

        if items_to_compute is not None:
            item_scores_user_similarity = - np.ones((len(user_id_array), self.URM_train.shape[1]),
                                                    dtype=np.float32) * np.inf
            item_scores_user_similarity_all = user_weights_array.dot(self.URM_train).toarray()
            item_scores_user_similarity[:, items_to_compute] = item_scores_user_similarity_all[:, items_to_compute]
        else:
            item_scores_user_similarity = user_weights_array.dot(self.URM_train)

        user_profile_array = item_scores_user_similarity + self.URM_train[user_id_array]

        if items_to_compute is not None:
            item_scores_item_similarity = - np.ones((len(user_id_array), self.URM_train.shape[1]),
                                                    dtype=np.float32) * np.inf
            item_scores_item_similarity_all = user_profile_array.dot(self.item_W_sparse).toarray()
            item_scores_item_similarity[:, items_to_compute] = item_scores_item_similarity_all[:, items_to_compute]
        else:
            item_scores_item_similarity = user_profile_array.dot(self.item_W_sparse).toarray()

        return item_scores_item_similarity

    def _check_format(self):
        if not self._URM_train_format_checked:

            if self.URM_train.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down the computation.".format(
                        "URM_train", "csr"))

            self._URM_train_format_checked = True

        if not self._item_W_sparse_format_checked:

            if self.item_W_sparse.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down "
                    "the computation.".format("item_W_sparse", "csr"))

            self._item_W_sparse_format_checked = True

        if not self._user_W_sparse_format_checked:

            if self.user_W_sparse.getformat() != "csr":
                self._print(
                    "PERFORMANCE ALERT compute_item_score: {} is not {}, this will significantly slow down "
                    "the computation.".format("user__sparse", "csr"))

            self._user_W_sparse_format_checked = True

    def save_model(self, folder_path, file_name=None):
        if file_name is None:
            file_name = self.RECOMMENDER_NAME

        self._print("Saving model in file '{}'".format(folder_path + file_name))

        data_dict_to_save = {"user_W_sparse": self.user_W_sparse, "item_W_sparse": self.item_W_sparse}

        dataIO = DataIO(folder_path=folder_path)
        dataIO.save_data(file_name=file_name, data_dict_to_save=data_dict_to_save)

        self._print("Saving complete")
