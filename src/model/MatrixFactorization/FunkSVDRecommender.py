import pandas as pd
import surprise
from surprise.prediction_algorithms import matrix_factorization

from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender

import numpy as np

class FunkSVDRecommender(BaseMatrixFactorizationRecommender):
    """ Funk SVD Recommender"""

    RECOMMENDER_NAME = "FunkSVDRecommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

        row = self.URM_train.tocoo().row
        col = self.URM_train.tocoo().col
        data = self.URM_train.tocoo().data
        df = pd.DataFrame(data={'userID': row, 'itemID': col, 'rating': data})
        reader = surprise.Reader(rating_scale=(0, 1))
        self.data = surprise.Dataset.load_from_df(df[['userID', 'itemID', 'rating']], reader)
        self.trainset = self.data.build_full_trainset()

    def fit(self, epochs=300, num_factors=50, regularization=0.01, learning_rate=0.01, implicit=False):
        self.model = matrix_factorization.SVD(n_factors=num_factors, n_epochs=epochs, biased=~implicit,
                                              lr_all=learning_rate, reg_all=regularization, verbose=True)

        self.model.fit(self.trainset)
        raw_uids = [self.trainset.to_raw_uid(uid) for uid in np.arange(self.trainset.n_users)]
        raw_iids = [self.trainset.to_raw_iid(uid) for uid in np.arange(self.trainset.n_items)]
        self.USER_factors = np.zeros(shape=(self.n_users, num_factors))
        self.ITEM_factors = np.zeros(shape=(self.n_items, num_factors))
        self.USER_factors[raw_uids, :] = self.model.pu
        self.ITEM_factors[raw_iids, :] = self.model.qi

        if ~implicit:
            self.ITEM_bias = np.zeros(shape=self.n_items)
            self.USER_bias = np.zeros(shape=self.n_users)
            self.GLOBAL_bias = self.trainset.global_mean

            self.ITEM_bias[raw_iids] = self.model.bi
            self.USER_bias[raw_uids] = self.model.bu
            self.use_bias = True