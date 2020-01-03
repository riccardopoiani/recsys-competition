import unittest

from sklearn.preprocessing import MinMaxScaler

from scripts.scripts_utils import read_split_load_data
from src.data_management.data_reader import get_ICM_train_new, get_UCM_train
from src.feature.demographics_content import get_user_demographic
from src.model import new_best_models
from src.model.Ensemble.Boosting.boosting_preprocessing import get_boosting_base_dataframe, get_label_array, \
    add_random_negative_ratings, get_train_dataframe_proportion, add_recommender_predictions, \
    advanced_subclass_handling, add_ICM_information, add_UCM_information, add_user_len_information, add_item_popularity

import numpy as np


class PreprocessingBoostingTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.k_out = 3
        self.cutoff = 5
        self.path = "../../data/"

        self.data_reader = read_split_load_data(self.k_out, allow_cold_users=False, seed=1000)

        self.URM_train, self.URM_test = self.data_reader.get_holdout_split()
        self.ICM_all, _ = get_ICM_train_new(self.data_reader)
        self.UCM_all = get_UCM_train(self.data_reader)

        self.main_rec = new_best_models.ItemCBF_CF.get_model(URM_train=self.URM_train, ICM_train=self.ICM_all)

    def test_get_boosting_base_dataframe(self):
        n_users = 1000  # Number of users to test
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        true_recommendations = np.array(self.main_rec.recommend(user_id_array=user_id_array, cutoff=self.cutoff,
                                                                remove_seen_flag=True))
        user_recommendations_items = true_recommendations.reshape((true_recommendations.size, 1)).squeeze()

        flag = False
        for i, user in enumerate(user_id_array):
            df_items = df['item_id'].iloc[i * self.cutoff: i * self.cutoff + self.cutoff].values
            true_items = user_recommendations_items[i * self.cutoff: i * self.cutoff + self.cutoff]
            if np.any(np.in1d(df_items, true_items, assume_unique=True)):
                flag = True
                break
        assert flag == False

    def test_get_label_array(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)

        labels = np.array(self.URM_train[df['user_id'].values, df['item_id'].values].tolist()).flatten()
        assert np.array_equal(labels, label_array)

    def test_add_random_negative_ratings(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        new_df = add_random_negative_ratings(data_frame=df, URM_train=self.URM_train, proportion=1)

        new_elements_id = np.arange(len(df), len(new_df))
        user_ids = new_df.iloc[new_elements_id]['user_id'].values
        item_ids = new_df.iloc[new_elements_id]['item_id'].values

        shifted_invalid_items = np.left_shift(item_ids, np.uint64(np.log2(n_users) + 1))
        tuple_user_item = np.bitwise_or(user_ids, shifted_invalid_items)
        unique_tuple = np.unique(tuple_user_item)
        assert unique_tuple.size == tuple_user_item.size

        labels = np.array(self.URM_train[user_ids, item_ids].tolist()).flatten()
        assert np.any(labels > 0) == False

    def test_add_recommender_predictions(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array
        df = add_random_negative_ratings(data_frame=df, URM_train=self.URM_train, proportion=1)

        # Need to reorder the dataframe in order for the add_recommender_predictions to work
        df = df.sort_values(by="user_id", ascending=True)
        df = df.reset_index()
        df = df.drop(columns=["index"], inplace=False)

        new_df = add_recommender_predictions(data_frame=df, recommender=self.main_rec,
                                             column_name=self.main_rec.RECOMMENDER_NAME)

        # Test that all scores are correct
        all_scores = self.main_rec._compute_item_score(user_id_array)
        scaler = MinMaxScaler()
        scaler.fit(all_scores.reshape(-1, 1))
        all_scores = np.reshape(scaler.transform(all_scores.reshape(-1, 1)), newshape=all_scores.shape)

        for i in range(len(new_df)):
            user = new_df['user_id'].iloc[i]
            item = new_df['item_id'].iloc[i]
            score = new_df[self.main_rec.RECOMMENDER_NAME].iloc[i]

            assert score == all_scores[user, item]

    def test_advanced_subclass_handling(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        new_df = advanced_subclass_handling(data_frame=df, URM_train=self.URM_train)

        print(new_df)
        print(new_df.columns)
        # TODO: test, but kind of difficult

    def test_add_ICM_information(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        new_df = add_ICM_information(df, self.path)

        print(new_df)
        # TODO: test, but kind of difficult

    def test_add_UCM_information(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        new_df = add_UCM_information(df, self.data_reader.get_original_user_id_to_index_mapper(), self.path)

        UCM_age = self.data_reader.get_UCM_from_name("UCM_age")
        age_mapper = self.data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
        age_demographic = get_user_demographic(UCM_age, age_mapper)

        UCM_region = self.data_reader.get_UCM_from_name("UCM_region")
        region_mapper = self.data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_region")
        id_to_original_region_mapper = {v: int(k) for k, v in region_mapper.items()}
        for i in range(len(new_df)):
            user = new_df['user_id'].iloc[i]

            # Test age
            age = new_df['age'].iloc[i]
            age_imputed_flag = new_df['age_imputed_flag'].iloc[i]

            if age_demographic[user] == -1:
                assert age_imputed_flag == 1
                assert age == 5  # Imputed value (mode + 1)
            else:
                assert age_imputed_flag == 0
                assert age == age_demographic[user]

            # Test region
            true_regions = UCM_region.indices[UCM_region.indptr[user]: UCM_region.indptr[user+1]]
            true_regions = [id_to_original_region_mapper[true_region] for true_region in true_regions]
            for region in id_to_original_region_mapper.values():
                column_name = "region_{}".format(region)
                region_in_newdf = new_df[column_name].iloc[i]

                if region in true_regions:
                    assert region_in_newdf == 1, "User {} has not correct region {}".format(user, region)
                else:
                    assert region_in_newdf == 0, "User {} has not correct region {}".format(user, region)

    def test_add_UCM_information_age_onehot(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        new_df = add_UCM_information(df, self.data_reader.get_original_user_id_to_index_mapper(), self.path,
                                     use_age_onehot=True)

        UCM_age = self.data_reader.get_UCM_from_name("UCM_age")
        age_mapper = self.data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
        id_to_original_age_mapper = {v: int(k) for k, v in age_mapper.items()}
        for i in range(len(new_df)):
            user = new_df['user_id'].iloc[i]
            # Test age
            ages_original = UCM_age.indices[UCM_age.indptr[user]: UCM_age.indptr[user + 1]]
            ages_original = [id_to_original_age_mapper[age] for age in ages_original]
            age_imputed_flag = new_df['age_imputed_flag'].iloc[i]

            for original_age in id_to_original_age_mapper.values():
                column_name = "age_{}".format(original_age)
                age_in_newdf = new_df[column_name].iloc[i]

                if original_age in ages_original:
                    assert age_in_newdf == 1
                elif age_imputed_flag == 1 and original_age == 5:
                    assert age_in_newdf == 1
                else:
                    assert age_in_newdf == 0, "User {} has incorrect age {}".format(user, original_age)

    def test_add_user_len_information(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        newdf = add_user_len_information(df, self.URM_train)

        for i in range(len(newdf)):
            user = newdf['user_id'].iloc[i]
            user_profile_len = newdf['user_act'].iloc[i]

            true_user_profile_len = len(self.URM_train.indices[self.URM_train.indptr[user]: self.URM_train.indptr[user+1]])

            assert user_profile_len == true_user_profile_len

    def test_add_item_popularity(self):
        n_users = 5000
        user_id_array = np.arange(n_users)

        df = get_boosting_base_dataframe(user_id_array, self.main_rec, cutoff=self.cutoff)
        label_array, _, _ = get_label_array(df, self.URM_train)
        df['label'] = label_array

        newdf = add_item_popularity(df, self.URM_train)
        URM_train_csc = self.URM_train.tocsc()
        for i in range(len(newdf)):
            item = newdf['item_id'].iloc[i]
            item_pop = newdf['item_pop'].iloc[i]

            true_item_pop = len(URM_train_csc.indices[URM_train_csc.indptr[item]: URM_train_csc.indptr[item+1]])

            assert item_pop == true_item_pop

    def test_get_train_dataframe_proportion(self):
        n_users = 500
        user_id_array = np.arange(n_users)

        df = get_train_dataframe_proportion(user_id_array, self.cutoff, self.main_rec, self.path,
                                            mapper=self.data_reader.get_original_user_id_to_index_mapper(),
                                            recommender_list=[self.main_rec], URM_train=self.URM_train, proportion=1)

        # Test that the df is ordered by user_id
        users = df['user_id'].values
        assert np.all(users[i] <= users[i + 1] for i in range(users.size - 1))

        # Test get_boosting_base_dataframe
        unique_users, user_indptr = np.unique(users, return_index=True)
        user_indptr = np.concatenate([user_indptr, [users.size]])
        true_recommendations = np.array(self.main_rec.recommend(user_id_array=user_id_array, cutoff=self.cutoff,
                                                                remove_seen_flag=True))
        user_recommendations_items = true_recommendations.reshape((true_recommendations.size, 1)).squeeze()

        flag = False
        for i, user in enumerate(user_id_array):
            df_items = df['item_id'].iloc[user_indptr[user]: user_indptr[user]].values
            true_items = user_recommendations_items[i * self.cutoff: i * self.cutoff + self.cutoff]
            if np.any(np.in1d(df_items, true_items, assume_unique=True)):
                flag = True
                break
        assert flag == False

        # Test labels value
        labels = np.array(self.URM_train[df['user_id'].values, df['item_id'].values].tolist()).flatten()
        assert np.array_equal(labels, df['label'].values)

        # Test recommender predictions
        all_scores = self.main_rec._compute_item_score(user_id_array)
        scaler = MinMaxScaler()
        scaler.fit(all_scores.reshape(-1, 1))
        all_scores = np.reshape(scaler.transform(all_scores.reshape(-1, 1)), newshape=all_scores.shape)

        for i in range(len(df)):
            user = df['user_id'].iloc[i]
            item = df['item_id'].iloc[i]
            score = df[self.main_rec.RECOMMENDER_NAME].iloc[i]

            assert score == all_scores[user, item]

        # Test advanced subclass

        # Test ICM information

        # Test UCM information
        UCM_age = self.data_reader.get_UCM_from_name("UCM_age")
        age_mapper = self.data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_age")
        age_demographic = get_user_demographic(UCM_age, age_mapper)

        UCM_region = self.data_reader.get_UCM_from_name("UCM_region")
        region_mapper = self.data_reader.dataReader_object.get_UCM_feature_to_index_mapper_from_name("UCM_region")
        id_to_original_region_mapper = {v: int(k) for k, v in region_mapper.items()}
        for i in range(len(df)):
            user = df['user_id'].iloc[i]

            # Test age
            age = df['age'].iloc[i]
            age_imputed_flag = df['age_imputed_flag'].iloc[i]

            if age_demographic[user] == -1:
                assert age_imputed_flag == 1
                assert age == 5  # Imputed value (mode + 1)
            else:
                assert age_imputed_flag == 0
                assert age == age_demographic[user]

            # Test region
            true_regions = UCM_region.indices[UCM_region.indptr[user]: UCM_region.indptr[user + 1]]
            true_regions = [id_to_original_region_mapper[true_region] for true_region in true_regions]
            for region in id_to_original_region_mapper.values():
                column_name = "region_{}".format(region)
                region_in_newdf = df[column_name].iloc[i]

                if region in true_regions:
                    assert region_in_newdf == 1, "User {} has not correct region {}".format(user, region)
                else:
                    assert region_in_newdf == 0, "User {} has not correct region {}".format(user, region)

        # Test user_activity
        for i in range(len(df)):
            user = df['user_id'].iloc[i]
            user_profile_len = df['user_act'].iloc[i]

            true_user_profile_len = len(self.URM_train.indices[self.URM_train.indptr[user]: self.URM_train.indptr[user+1]])

            assert user_profile_len == true_user_profile_len

        # Test item_popularity
        URM_train_csc = self.URM_train.tocsc()
        for i in range(len(df)):
            item = df['item_id'].iloc[i]
            item_pop = df['item_pop'].iloc[i]

            true_item_pop = len(URM_train_csc.indices[URM_train_csc.indptr[item]: URM_train_csc.indptr[item + 1]])

            assert item_pop == true_item_pop




if __name__ == '__main__':
    unittest.main()
