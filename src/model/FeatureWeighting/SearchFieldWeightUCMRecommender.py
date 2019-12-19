from course_lib.Base.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps


class SearchFieldWeightUCMRecommender(BaseRecommender):
    """ Search Field Weight UCM Recommender """

    RECOMMENDER_NAME = "SearchFieldWeightUCMRecommender"

    def __init__(self, URM_train, UCM_train, recommender_class: classmethod, recommender_par: dict,
                 user_feature_to_cols_mapper: dict, verbose=True):
        super(SearchFieldWeightUCMRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommender_class = recommender_class
        self.recommender_par = recommender_par
        self.user_feature_to_cols_mapper = user_feature_to_cols_mapper
        self.UCM_train: sps.csr_matrix = UCM_train
        self.model = None

    def fit(self, **field_weights):
        user_feature_weights = np.ones(shape=self.UCM_train.shape[1])
        for feature_name, weight in field_weights.items():
            start, end = self.user_feature_to_cols_mapper[feature_name]
            user_feature_weights[start:end] = user_feature_weights[start:end]*weight
        user_feature_weights_diag = sps.diags(user_feature_weights)
        self.UCM_train = self.UCM_train.dot(user_feature_weights_diag)

        self.model = self.recommender_class(self.URM_train, self.UCM_train)
        self.model.fit(**self.recommender_par)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        return self.model._compute_item_score(user_id_array=user_id_array, items_to_compute=items_to_compute)

    def save_model(self, folder_path, file_name=None):
        pass
