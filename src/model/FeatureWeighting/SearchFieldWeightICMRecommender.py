from course_lib.Base.BaseRecommender import BaseRecommender
import numpy as np
import scipy.sparse as sps


class SearchFieldWeightICMRecommender(BaseRecommender):
    """ Search Field Weight ICM Recommender """

    RECOMMENDER_NAME = "SearchFieldWeightICMRecommender"

    def __init__(self, URM_train, ICM_train, recommender_class: classmethod, recommender_par: dict,
                 item_feature_to_range_mapper: dict, verbose=True):
        super(SearchFieldWeightICMRecommender, self).__init__(URM_train, verbose=verbose)

        self.recommender_class = recommender_class
        self.recommender_par = recommender_par
        self.item_feature_to_range_mapper = item_feature_to_range_mapper
        self.ICM_train: sps.csr_matrix = ICM_train
        self.model = None

    def fit(self, **field_weights):
        item_feature_weights = np.ones(shape=self.ICM_train.shape[1])
        for feature_name, weight in field_weights.items():
            start, end = self.item_feature_to_range_mapper[feature_name]
            item_feature_weights[start:end] = item_feature_weights[start:end]*weight
        user_feature_weights_diag = sps.diags(item_feature_weights)
        self.ICM_train = self.ICM_train.dot(user_feature_weights_diag)

        self.model = self.recommender_class(self.URM_train, self.ICM_train)
        self.model.fit(**self.recommender_par)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        return self.model._compute_item_score(user_id_array=user_id_array, items_to_compute=items_to_compute)

    def save_model(self, folder_path, file_name=None):
        pass
