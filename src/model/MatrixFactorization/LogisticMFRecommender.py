from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import implicit


class LogisticMFRecommender(BaseMatrixFactorizationRecommender):
    """ Logistic MF Recommender """

    RECOMMENDER_NAME = "LogisticMFRecommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

    def fit(self, epochs=300, num_factors=50, regularization=0.01, neg_prop=30, learning_rate=0.001):
        self.model = implicit.lmf.LogisticMatrixFactorization(factors=num_factors, regularization=regularization,
                                                              learning_rate=learning_rate, neg_prop=neg_prop,
                                                              iterations=epochs)
        self.model.fit(self.URM_train.T.tocsr(), show_progress=True)
        self.USER_factors = self.model.user_factors
        self.ITEM_factors = self.model.item_factors
