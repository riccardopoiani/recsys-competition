import xgboost as xgb
from sklearn.model_selection import train_test_split

from src.model.Ensemble.Boosting.boosting_preprocessing import *
from src.utils.general_utility_functions import get_split_seed


class BoostingFixedData(BaseRecommender):

    def __init__(self, URM_train, X, y, df_test, cutoff, valid_size=0.2):
        super().__init__(URM_train)

        X_train, X_valid, y_train, y_valid = train_test_split(X.drop(columns=["user_id", "item_id"], inplace=False),
                                                              y, test_size=valid_size, random_state=get_split_seed())

        self.dtrain = xgb.DMatrix(data=X_train, label=y_train)
        self.dvalid = xgb.DMatrix(data=X_valid, label=y_valid)
        self.evallist = [(self.dtrain, 'train'), (self.dvalid, 'eval')]
        self.bst = None
        self.dict_result = dict()
        self.dftest = df_test
        self.cutoff = cutoff
        self.items = np.ediff1d(self.URM_train.tocsc().indptr)
        self.loaded_from_file = False

    def train(self, num_round, param, early_stopping_round):
        self.bst = xgb.train(params=param, dtrain=self.dtrain, num_boost_round=num_round, evals=self.evallist,
                             early_stopping_rounds=early_stopping_round, evals_result=self.dict_result)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        if user_id_array[0] == 16391:
            debug = True
        else:
            debug = False

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
        dpredict = xgb.DMatrix(data_frame.drop(columns=["user_id", "item_id"], inplace=False))

        if debug:
            print(dpredict)

        if self.loaded_from_file:
            y_hat = self.bst.predict(dpredict)
        else:
            y_hat = self.bst.predict(dpredict, ntree_limit=self.bst.best_ntree_limit)

        scores = np.zeros(shape=(user_id_array.size, all_items.size))

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
