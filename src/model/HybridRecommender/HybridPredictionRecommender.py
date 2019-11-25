from course_lib.Base.BaseRecommender import BaseRecommender
from typing import List
import numpy as np


class HybridPredictionRecommenderDebug(BaseRecommender):
    models_object: List[BaseRecommender] = []
    models_name: List[str] = []
    models_aps: List[np.array] = []

    def __init__(self, URM_train):
        self.model_ap_size = None
        self.models_to_be_used = None
        super().__init__(URM_train)

    def add_fitted_model(self, recommender_name: str, recommender_object: BaseRecommender, recommender_aps: np.array):
        '''
        Add an already fitted model to the list of model that will be used to compute predictions.
        Models are assumed to be fitted on the same URM_train and validated on the same URM_validation.
        Also, recommendation average precision, are assumed to refer to the same map user-index.

        :param recommender_name: name of the recommender
        :param recommender_object: fitted recommended
        :param recommender_aps: average precision of the recommender on the validation set
        :return: None
        '''
        if not (self.__verify_aps_consistency__(recommender_aps) and self.__verify_name_consistency__(
                recommender_name)):
            raise AssertionError("The len of the aps of each recommender should be the same. Moreover, the name"
                                 "should not be in the ones already used")
        if len(self.models_name) == 0:
            self.model_ap_size = recommender_aps.size
        self.models_name.append(recommender_name)
        self.models_object.append(recommender_object)
        self.models_aps.append(recommender_aps)

    def get_number_of_models(self):
        return len(self.models_name)

    def get_recommender_names(self):
        return self.models_name

    def __verify_name_consistency__(self, name):
        return False if name in self.models_name else True

    def __verify_aps_consistency__(self, aps):
        '''
        Verify that each recommender has the same number of tested recommendations
        :param aps: average precision to be checked
        :return: True if condition are satisfied, False otherwise
        '''
        if len(self.models_aps) == 0:
            return True
        return True if (self.model_ap_size == aps.size) else False

    def get_model_to_be_used(self):
        if self.models_to_be_used is None:
            return self.models_to_be_used
        else:
            raise RuntimeError("You need to fit the recommender first")

    def fit(self):
        '''
        Compute for each user predicted by the recommenders, the recommender with the highest MAP.

        We have a huge list (around 50k) for each recommender stored.
        We need to select index of the recommender associated to the highest value in these list, for
        each position.

        We could get the max of the prediction for each component (i.e. the nparray containing all the maximum values).
        -> To do that, we should transform, all the values to a matrix, and take the max on the second axis.

        After that, we should build a mask, doing checks on them, and finally, break ties (if any)


        :return: None
        '''
        # Getting the maximum value
        print("Retrieving max values...", end="")
        matrix = np.array(self.models_aps)
        max_values = matrix.max(axis=0)  # np array containing the maximum values
        print("Done")

        print("Building masks...", end="")
        # Building the masks
        masks = []
        for aps in self.models_aps:
            res = np.where(aps == max_values, 1, 0)
            masks.append(res)
        mask_matrix = np.array(masks)
        print("Done")

        print("Computing model to be used...")
        # Now, that we know that, we should build self.model_to_be_used
        self.models_to_be_used = mask_matrix.argmax(axis=0)
        print("Done")

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_CustomItems_flag=False, return_scores=False):
        recommendations = []

        junk, scores = self.models_object[0].recommend(user_id_array, cutoff=cutoff,
                                                       remove_CustomItems_flag=remove_CustomItems_flag,
                                                       items_to_compute=items_to_compute,
                                                       remove_top_pop_flag=remove_top_pop_flag,
                                                       return_scores=True)

        # Building recommendations and scores
        for i in range(len(user_id_array)):
            rec_idx = self.models_to_be_used[user_id_array[i]]
            if return_scores:
                recommendation_for_user = self.models_object[rec_idx].recommend(user_id_array[i],
                                                                                cutoff=cutoff,
                                                                                remove_CustomItems_flag=remove_CustomItems_flag,
                                                                                items_to_compute=items_to_compute,
                                                                                remove_top_pop_flag=remove_top_pop_flag,
                                                                                return_scores=False)
            else:
                recommendation_for_user = self.models_object[rec_idx].recommend(user_id_array[i], cutoff=cutoff,
                                                                                remove_CustomItems_flag=remove_CustomItems_flag,
                                                                                items_to_compute=items_to_compute,
                                                                                remove_top_pop_flag=remove_top_pop_flag,
                                                                                return_scores=False)

            recommendations.append(recommendation_for_user)

        # Return predictions
        if return_scores:
            return recommendations, scores
        else:
            return recommendations