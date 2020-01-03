import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.model.Ensemble.Boosting.boosting_preprocessing import *


class BoostingFixedData(BaseRecommender):

    def __init__(self, URM_train, X, y, df_test, cutoff, valid_size=0.2, URM_test=None, random_state=None):
        super().__init__(URM_train)

        X_train, X_valid, y_train, y_valid = train_test_split(X.drop(columns=["item_id"], inplace=False),
                                                              y, test_size=valid_size, random_state=random_state)

        """group_indices = np.argsort(X_train["user_id"].values)
        group_labels = X_train["user_id"].values[group_indices]
        _, sizes = np.unique(group_labels, return_counts=True)
        X_train = X_train.iloc[group_indices].drop(columns=["user_id"], inplace=False)
        y_train = y_train[group_indices]
        self.dtrain = xgb.DMatrix(data=X_train, label=y_train)
        self.dtrain.set_group(sizes)

        group_indices = np.argsort(X_valid["user_id"].values)
        group_labels = X_valid["user_id"].values[group_indices]
        _, sizes = np.unique(group_labels, return_counts=True)
        X_valid = X_valid.iloc[group_indices].drop(columns=["user_id"], inplace=False)
        y_valid = y_valid[group_indices]
        self.dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        self.dvalid.set_group(sizes)"""

        group_labels = X["user_id"].values
        _, sizes = np.unique(group_labels, return_counts=True)
        X = X.drop(columns=["user_id", "item_id"], inplace=False)
        self.dtrain = xgb.DMatrix(data=X, label=y)
        self.dtrain.set_group(sizes)

        group_labels = df_test["user_id"].values
        _, sizes = np.unique(group_labels, return_counts=True)
        y_test, _, _ = get_label_array(df_test, URM_train + URM_test)
        X_test = df_test.drop(columns=["user_id", "item_id"], inplace=False)
        self.dvalid = xgb.DMatrix(data=X_test, label=y_test)
        self.dvalid.set_group(sizes)

        self.evallist = [(self.dtrain, 'train'), (self.dvalid, 'eval')]
        self.bst = None
        self.dict_result = dict()
        self.dftest = df_test
        self.cutoff = cutoff
        self.items = np.ediff1d(self.URM_train.tocsc().indptr)
        self.loaded_from_file = False

    def train(self, num_round, param, early_stopping_round):
        self.bst = xgb.train(params=param, dtrain=self.dtrain, num_boost_round=num_round, evals=self.evallist,
                             early_stopping_rounds=early_stopping_round, evals_result=self.dict_result,
                             maximize=True)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        user_id_array_sorted = np.sort(user_id_array)
        idx = np.searchsorted(user_id_array_sorted, user_id_array)
        user_id_array = user_id_array_sorted

        # Setting total items: the one of which you have to compute scores
        if items_to_compute is not None:
            raise NotImplemented("Feature not implemented yet")
        else:
            all_items = self.items.copy()

        # Get the test dataframe
        data_frame = self.dftest.copy()
        data_frame = data_frame[data_frame['user_id'].isin(user_id_array)]

        # Predict the ratings
        items = data_frame["item_id"].values
        group_labels = data_frame["user_id"].values
        _, sizes = np.unique(group_labels, return_counts=True)
        X_test = data_frame.drop(columns=["user_id", "item_id"], inplace=False)
        dpredict = xgb.DMatrix(X_test)
        dpredict.set_group(sizes)

        if self.loaded_from_file:
            y_hat = self.bst.predict(dpredict)
        else:
            y_hat = self.bst.predict(dpredict, ntree_limit=self.bst.best_ntree_limit)

        scores = np.zeros(shape=(user_id_array.size, self.n_items))

        for i, user in enumerate(user_id_array):
            # Getting the item of this users that should be re-ranked
            curr_items = np.array(items[(i * self.cutoff):(i * self.cutoff) + self.cutoff])

            # Getting the predictions for this users, and sort them
            pred_for_user = y_hat[(i * self.cutoff):(i * self.cutoff) + self.cutoff]

            scores[i][curr_items] = pred_for_user

        # Resort scores according to the original index
        return scores[idx]

    def load_model_from_file(self, file_path, params):
        self.loaded_from_file = True
        self.bst = xgb.Booster(params)
        self.bst.load_model(file_path)

    def save_model(self, folder_path, file_name=None):
        self.bst.save_model(folder_path + file_name)
