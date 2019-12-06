import os

import numpy as np
import xlearn as xl

from course_lib.Base.BaseRecommender import BaseRecommender
from src.data_management.data_preprocessing_fm import format_URM_slice_uncompressed


class FactorizationMachineRecommender(BaseRecommender):
    """ Factorization Machine Recommender """

    RECOMMENDER_NAME = "FactorizationMachineRecommender"

    def __init__(self, URM_train, train_svm_file_path, approximate_recommender: BaseRecommender, ICM_train=None,
                 max_items_to_predict=1000, model_path="./model.out", temp_folder="temp/", verbose=True):
        self.ICM_train = ICM_train
        self.approximate_recommender = approximate_recommender
        self.max_items_to_predict = max_items_to_predict

        self.temp_folder = temp_folder
        self.model_path = model_path
        self.model = xl.create_fm()
        self.model.setTrain(train_svm_file_path)

        super().__init__(URM_train, verbose)

    def fit(self, epochs=300, latent_factors=100, regularization=0.01, learning_rate=0.01, optimizer="adagrad"):
        if not os.path.exists(self.temp_folder):
            os.makedirs(self.temp_folder)

        params = {'task': 'binary', 'epoch': epochs, 'k': latent_factors, 'lambda': regularization, 'opt': optimizer}
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
        items_to_recommend = self.get_items_to_recommend(user_id_array, n_items)
        recommendation_file = os.path.join(self.temp_folder,
                                           "recommendations_{}_{}.txt".format(user_id_array[0], user_id_array[-1]))
        if not os.path.isfile(recommendation_file):
            if self.ICM_train is None:
                FM_matrix = format_URM_slice_uncompressed(user_id_array, items_to_recommend, self.URM_train.shape[0])
                labels = np.ones(shape=FM_matrix.shape[0])
                xl.dump_svmlight_file(X=FM_matrix, y=labels,
                                      f=recommendation_file)
            else:
                # TODO
                pass
        self.model.setSigmoid()
        self.model.setTest(recommendation_file)

        scores_batch = np.reshape(self.model.predict(model_path=self.model_path), newshape=items_to_recommend.shape)
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

    def get_items_to_recommend(self, user_id_array, n_items):
        return np.array(self.approximate_recommender.recommend(user_id_array, cutoff=n_items))
