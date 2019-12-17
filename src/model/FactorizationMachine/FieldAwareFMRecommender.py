import os

import numpy as np
import xlearn as xl
from xlearn import write_data_to_xlearn_format

from course_lib.Base.BaseRecommender import BaseRecommender
from src.data_management.data_preprocessing_fm import format_URM_slice_uncompressed, add_ICM_info, add_UCM_info
from src.utils.general_utility_functions import get_project_root_path


class FieldAwareFMRecommender(BaseRecommender):
    """ Field Aware FM Recommender"""

    RECOMMENDER_NAME = "FieldAwareFMRecommender"

    def __init__(self, URM_train, train_svm_file_path, approximate_recommender: BaseRecommender,
                 ICM_train=None, UCM_train=None, item_feature_fields=None, user_feature_fields=None,
                 valid_svm_file_path=None, max_items_to_predict=1000, model_filename="model.out",
                 temp_relative_folder="temp/", verbose=True):
        self.ICM_train = ICM_train
        self.UCM_train = UCM_train
        user_fields = np.full(shape=URM_train.shape[0], fill_value=0)
        item_fields = np.full(shape=URM_train.shape[1], fill_value=1)
        if item_feature_fields is not None:
            item_feature_fields = item_feature_fields + 2
        if user_feature_fields is not None:
            user_feature_fields = user_feature_fields + np.max(item_feature_fields) + 1
        self.fields = np.concatenate([user_fields, item_fields, item_feature_fields, user_feature_fields])

        self.approximate_recommender = approximate_recommender
        self.max_items_to_predict = max_items_to_predict

        # Set path of temp folder and model_path
        root_path = get_project_root_path()
        fm_data_path = os.path.join(root_path, "resources", "ffm_data")
        self.temp_folder = os.path.join(fm_data_path, temp_relative_folder)
        self.model_folder = os.path.join(fm_data_path, "model")
        self.model_path = os.path.join(self.model_folder, model_filename)

        self.model = xl.create_ffm()
        self.model.setTrain(train_svm_file_path)
        if valid_svm_file_path is not None:
            self.model.setValidate(valid_svm_file_path)

        super().__init__(URM_train, verbose)

    def load_pre_model(self, pre_model_path):
        self.model.setPreModel(pre_model_path)

    def fit(self, epochs=300, latent_factors=100, regularization=0.01, learning_rate=0.01, optimizer="adagrad",
            stop_window=10, metric=None):
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

        if not os.path.exists(self.model_folder):
            os.makedirs(self.model_folder)

        params = {'task': 'binary', 'epoch': epochs, 'k': latent_factors, 'lambda': regularization, 'opt': optimizer,
                  'stop_window': stop_window, 'lr': learning_rate}
        if metric is not None:
            params["metric"] = metric
        self.model.fit(params, model_path=self.model_path)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        if np.isscalar(user_id_array):
            user_id_array = np.atleast_1d(user_id_array)
            single_user = True
        else:
            single_user = False

        if cutoff is None:
            cutoff = self.URM_train.shape[1] - 1

        n_items = self.max_items_to_predict
        items_to_recommend = self._get_items_to_recommend(user_id_array, n_items)
        if self.ICM_train is None and self.UCM_train is None:
            recommendation_file = os.path.join(self.temp_folder,
                                               "recommendations_{}_{}.txt".format(user_id_array[0], user_id_array[-1]))
        else:
            recommendation_file = os.path.join(self.temp_folder,
                                               "recommendations_with_ICM_{}_{}.txt".format(user_id_array[0],
                                                                                           user_id_array[-1]))
        if not os.path.isfile(recommendation_file):
            fm_matrix = format_URM_slice_uncompressed(user_id_array, items_to_recommend, self.URM_train.shape[0],
                                                      self.URM_train.shape[0] + self.URM_train.shape[1])

            # First ICM, then UCM just like in the creation on the training set
            if self.ICM_train is not None:
                fm_matrix = add_ICM_info(fm_matrix, self.ICM_train, self.URM_train.shape[0])
            if self.UCM_train is not None:
                fm_matrix = add_UCM_info(fm_matrix, self.UCM_train, 0)
            fm_matrix = fm_matrix[:, :]
            labels = np.ones(shape=fm_matrix.shape[0])
            write_data_to_xlearn_format(X=fm_matrix, y=labels, filepath=recommendation_file, fields=self.fields)
        self.model.setSigmoid()
        self.model.setTest(recommendation_file)

        prediction_file = os.path.join(self.model_folder, "prediction.txt")
        self.model.predict(model_path=self.model_path, out_path=prediction_file)
        scores_batch = np.reshape(self.model.predict(model_path=self.model_path), newshape=(items_to_recommend.shape[0],
                                                                                            items_to_recommend.shape[1]))
        relevant_items_partition = (-scores_batch).argpartition(cutoff, axis=1)[:, 0:cutoff]
        relevant_items_partition_original_value = scores_batch[
            np.arange(scores_batch.shape[0])[:, None], relevant_items_partition]
        relevant_items_partition_sorting = np.argsort(-relevant_items_partition_original_value, axis=1)
        score_index_list = relevant_items_partition[
            np.arange(relevant_items_partition.shape[0])[:, None], relevant_items_partition_sorting]
        ranking_list = items_to_recommend[np.arange(scores_batch.shape[0]), np.transpose(score_index_list)].T

        if single_user:
            ranking_list = ranking_list[0]

        if return_scores:
            return ranking_list, np.empty(shape=(len(user_id_array), self.n_items))

        else:
            return ranking_list

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        # Useless function: avoided in order to do approximate recommendation
        pass

    def save_model(self, folder_path, file_name=None):
        # Useless function
        pass

    def _get_items_to_recommend(self, user_id_array, n_items):
        return np.array(self.approximate_recommender.recommend(user_id_array, cutoff=n_items))
