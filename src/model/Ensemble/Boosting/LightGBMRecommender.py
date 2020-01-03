from src.model.Ensemble.Boosting.boosting_preprocessing import *
import lightgbm as lgb


class LightGBMRecommender(BaseRecommender):

    RECOMMENDER_NAME = "LightGBMRecommender"

    def __init__(self, URM_train, X_train, y_train, X_test, y_test, cutoff_test, categorical_feature=None):
        super().__init__(URM_train)

        group_labels = X_train["user_id"].values
        _, sizes = np.unique(group_labels, return_counts=True)
        self.dtrain = lgb.Dataset(data=X_train.drop(columns=["user_id", "item_id"], inplace=False), label=y_train,
                                  categorical_feature=categorical_feature,
                                  group=sizes)

        group_labels = X_test["user_id"].values
        _, sizes = np.unique(group_labels, return_counts=True)
        self.dvalid = lgb.Dataset(data=X_test.drop(columns=["user_id", "item_id"], inplace=False), label=y_test,
                                  categorical_feature=categorical_feature,
                                  group=sizes)

        self.bst = None
        self.dict_result = dict()
        self.df_test = X_test
        self.cutoff = cutoff_test
        self.loaded_from_file = False

    def fit(self, num_iteration=1000, learning_rate=0.01, reg_l1=0, reg_l2=1, num_leaves=31, max_depth=4,
            min_gain_to_split=0.1, min_data_in_leaf=10, bagging_fraction=0.8, bagging_freq=16, feature_fraction=0.8,
            objective="lambdarank", metric="map", eval_at=10, max_position=10, early_stopping_round=None,
            verbose=True):

        parameters = {
            "learning_rate": learning_rate,  # Fixed learning rate and then decrease it if necessary on the future
            "min_gain_to_split": min_gain_to_split,  # min loss required to split a leaf
            "lambda_l1": reg_l1,  # regularizer L1
            "lambda_l2": reg_l2,  # regularizer L2
            "max_depth": max_depth,  # the larger, the higher prob. to overfitting
            "min_data_in_leaf": min_data_in_leaf,
            "bagging_fraction": bagging_fraction,
            "bagging_freq": bagging_freq,
            "feature_fraction": feature_fraction,
            "num_leaves": num_leaves,  # max number of leaves in a tree
            "objective": objective,  # Objective function to be optimized
            "metric": metric,
            "eval_at": [eval_at],
            "max_position": max_position,
            "is_unbalance": True
        }
        self.bst = lgb.train(params=parameters, train_set=self.dtrain, num_boost_round=num_iteration,
                             valid_sets=[self.dvalid, self.dtrain], valid_names=["valid", "train"],
                             early_stopping_rounds=early_stopping_round,
                             verbose_eval=verbose)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_id_array_sorted = np.sort(user_id_array)
        idx = np.searchsorted(user_id_array_sorted, user_id_array)
        user_id_array = user_id_array_sorted

        # Get the test dataframe
        data_frame = self.df_test.copy()
        data_frame = data_frame[data_frame['user_id'].isin(user_id_array)]

        # Predict the ratings
        items = data_frame["item_id"].values
        X_test = data_frame.drop(columns=["user_id", "item_id"])

        y_hat = self.bst.predict(X_test)

        scores = np.zeros(shape=(user_id_array.size, self.n_items))

        for i, user in enumerate(user_id_array):
            # Getting the item of this users that should be re-ranked
            curr_items = np.array(items[(i * self.cutoff):(i * self.cutoff) + self.cutoff])

            # Getting the predictions for this users, and sort them
            pred_for_user = y_hat[(i * self.cutoff):(i * self.cutoff) + self.cutoff]

            scores[i][curr_items] = pred_for_user

        # Resort scores according to the original index
        return scores[idx]

    def load_model(self, folder_path, file_name=None):
        self.loaded_from_file = True
        self.bst = lgb.Booster(model_file=folder_path + file_name)

    def save_model(self, folder_path, file_name=None):
        self.bst.save_model(folder_path + file_name + ".bin")
