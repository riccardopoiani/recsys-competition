from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import implicit


class MF_BPR_Recommender(BaseMatrixFactorizationRecommender):
    """ MF BPR Recommender """

    RECOMMENDER_NAME = "MF_BPR_Recommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

    def fit(self, epochs=300, num_factors=50, regularization=0.01, learning_rate=0.001):
        self.model = implicit.bpr.BayesianPersonalizedRanking(factors=num_factors, regularization=regularization,
                                                              learning_rate=learning_rate,
                                                              iterations=epochs, verify_negative_samples=True)
        self.model.fit(self.URM_train.T.tocsr())
        self.USER_factors = self.model.user_factors
        self.ITEM_factors = self.model.item_factors