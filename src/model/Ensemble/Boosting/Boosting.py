from course_lib.Base.BaseRecommender import BaseRecommender
from sklearn.model_selection import train_test_split
from src.utils.general_utility_functions import get_split_seed
from src.model.Ensemble.Boosting.boosting_preprocessing import *
import xgboost as xgb
import numpy as np
import pandas as pd


class Boosting(BaseRecommender):

    @staticmethod
    def get_dataframe(user_id_array, remove_seen_flag, cutoff, main_recommender, path, mapper, recommender_list,
                      URM_train):
        # Get dataframe for these users
        data_frame = get_boosting_base_dataframe(user_id_array=user_id_array, exclude_seen=remove_seen_flag,
                                                 cutoff=cutoff, top_recommender=main_recommender)
        for rec in recommender_list:
            data_frame = add_recommender_predictions(data_frame=data_frame, recommender=rec,
                                                     cutoff=cutoff, column_name=rec.RECOMMENDER_NAME)

        data_frame = add_ICM_information(data_frame=data_frame, path=path)
        data_frame = add_UCM_information(data_frame=data_frame, path=path, user_mapper=mapper)
        data_frame = add_user_len_information(data_frame=data_frame, URM_train=URM_train)

        return data_frame

    def __init__(self, URM_train, main_recommender: BaseRecommender, recommender_list: list, mapper: dict,
                 cutoff=20, test_size=0.2, path="../../data/"):
        """
        Initialize the method

        :param URM_train: URM on which the model is trained
        :param x: training data (taken from ICM)
        :param y: data label
        :param test_size:
        """
        super().__init__(URM_train)

        self.main_recommender = main_recommender
        self.recommender_list = recommender_list
        self.cutoff = cutoff
        self.path = path
        self.mapper = mapper

        x = Boosting.get_dataframe(user_id_array=URM_train.shape[0], remove_seen_flag=False,
                                   cutoff=self.cutoff, main_recommender=self.main_recommender,
                                   path=self.path, mapper=self.mapper, recommender_list=self.recommender_list,
                                   URM_train=URM_train)
        y = add_label(x, URM_train)

        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=get_split_seed())
        self.dtrain = xgb.DMatrix(data=X_train, label=y_train)
        self.dtest = xgb.DMatrix(data=X_test, label=y_test)
        self.evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]
        self.bst = None

    def train(self, num_round, param, early_stopping_round):
        self.bst = xgb.train(params=param, dtrain=self.dtrain, num_boost_round=num_round, evals=self.evallist,
                             early_stopping_rounds=early_stopping_round)

    def recommend(self, user_id_array, cutoff=None, remove_seen_flag=True, items_to_compute=None,
                  remove_top_pop_flag=False, remove_custom_items_flag=False, return_scores=False):
        data_frame = Boosting.get_dataframe(user_id_array=user_id_array, remove_seen_flag=remove_seen_flag,
                                            cutoff=self.cutoff, main_recommender=self.main_recommender,
                                            path=self.path, mapper=self.mapper, recommender_list=self.recommender_list,
                                            URM_train=self.URM_train)

        # Predict the ratings
        items = data_frame["item_id"]
        dpredict = xgb.DMatrix(data_frame)
        y_hat = self.bst.predict(dpredict)

        prediction_list = []

        for i, user in enumerate(user_id_array):
            # Getting the item of this users that should be re-ranked
            curr_items = np.array(items[(user * self.cutoff):(user * self.cutoff) + self.cutoff])

            # Getting the predictions for this users, and sort them
            pred_for_user = y_hat[i:i + self.cutoff]
            idx_rerank_item = np.argsort(pred_for_user)[-cutoff:]

            # Getting the items ranked correctly
            pred = curr_items[idx_rerank_item].tolist()

            prediction_list.append(pred)

        return prediction_list
