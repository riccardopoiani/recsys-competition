from course_lib.Base.BaseRecommender import BaseRecommender
from src.data_management.data_reader import get_users_of_age
from src.feature.demographics_content import get_user_demographic, get_sub_class_content

import numpy as np

class FilterRecommender(BaseRecommender):

    def __init__(self, URM_train, UCM_age, ICM_subclass, subclass_feature_to_id_mapper,
                 age_mapper_to_original, recommender: BaseRecommender, rerank_top_n=10):
        # Data
        self.URM = URM_train

        # Retrieving age information
        self.age_demographic = get_user_demographic(UCM_age, age_mapper_to_original, binned=True)
        age_list = np.sort(np.array(list(age_mapper_to_original.keys())))
        self.age_list = [int(age) for age in age_list]

        # Subclass information
        self.subclass_content_dict = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=True)
        self.subclass_content = get_sub_class_content(ICM_subclass, subclass_feature_to_id_mapper, binned=False)

        # Age-Subclass
        self.sub_age_dict = {}
        self.count_sub_ace_dict = {}

        # Inner recommender
        self.inner_recommender = recommender
        self.rerank_top_n = rerank_top_n

        # Recommender parameters
        self.filter_subclass_age = None

        self.filter_subclass_user = None
        self.min_num_ratings_subclass_user = None
        self.users_subclass = np.array([])

        self.subclass_rerank = None
        self.min_num_ratings_subclass_rerank = None
        self.max_ratings_user_subclass_rerank = None

        self.filter_price_per_user = None
        self.filter_asset_per_user = None
        self.filter_price_per_age = None
        self.filter_asset_per_age = None

        super().__init__(URM_train)

    def fit(self, filter_subclass_age=True, filter_subclass_user=True, min_num_ratings_subclass_user=50,
            subclass_rerank=False, min_ratings_user_subclass_rerank=50, max_ratings_user_subclass_rerank=100,
            filter_price_per_user=False, filter_asset_per_user=False, filter_price_per_age=False,
            filter_asset_per_age=False):
        # Parameters setting
        self.filter_subclass_age = filter_subclass_age
        self.filter_subclass_user = filter_subclass_user
        self.min_num_ratings_subclass_user = min_num_ratings_subclass_user
        self.subclass_rerank = subclass_rerank
        self.min_num_ratings_subclass_rerank = min_ratings_user_subclass_rerank
        self.max_ratings_user_subclass_rerank = max_ratings_user_subclass_rerank
        self.filter_price_per_user = filter_price_per_user
        self.filter_asset_per_user = filter_asset_per_user
        self.filter_price_per_age = filter_price_per_age
        self.filter_asset_per_age = filter_asset_per_age

        # User satisfying subclass conditions of minimum number of ratings
        mask = np.ediff1d(self.URM_train.tocsr().indptr) > min_num_ratings_subclass_user
        self.users_subclass = np.arange(self.URM_train.shape[0])[mask]

        # Mixing age and subclass information
        print("Mix age and subclass information...", end="")
        for age in self.age_list:
            users_age = get_users_of_age(age_demographic=self.age_demographic, age_list=[age])
            URM_age = self.URM_train[users_age].copy()
            items_age = URM_age.indices
            subclass_age = self.subclass_content[items_age]
            self.sub_age_dict[age] = subclass_age

        for age in self.age_list:
            curr_count = np.zeros(self.subclass_content.max())
            curr_sub_age = self.sub_age_dict[age]
            for i in range(curr_count.size):
                mask = np.in1d(curr_sub_age, [i])
                curr_count[i] = curr_sub_age[mask].size

            self.count_sub_ace_dict[age] = curr_count
        print("Done")

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        scores = self.inner_recommender._compute_item_score(user_id_array=user_id_array,
                                                            items_to_compute=items_to_compute)

        # Filter scores for items that are never bought from users of the subclass the users is in
        if self.filter_subclass_age:
            pass

        # Filter scores of items of subclasses that are never bought from a certain user (applied only if profile >
        # min_num_ratings_subclass_user)
        if self.filter_subclass_user:
            mask = np.in1d(user_id_array, self.users_subclass)
            temp_users = user_id_array[mask]  # Users affected by this filter
            # temp_URM = self.URM_train[temp_users].copy()

            for u in temp_users:
                items_u = self.URM_train[u].indices


        # rerank_top_n items according to user subclass preference (applied only on profile [min, max])
        if self.subclass_rerank:
            pass

        # Filter with user mean price
        if self.filter_price_per_user:
            pass

        # Filter with user mean asset
        if self.filter_asset_per_user:
            pass

        # Filter price per group of users
        if self.filter_asset_per_age:
            pass

        # Filter price per group of age
        if self.filter_price_per_age:
            pass

        return scores

    def save_model(self, folder_path, file_name=None):
        raise NotImplemented()
