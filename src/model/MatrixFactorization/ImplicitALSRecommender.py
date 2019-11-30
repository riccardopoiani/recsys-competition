from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
import implicit


class ImplicitALSRecommender(BaseMatrixFactorizationRecommender):
    """ Implicit ALS Recommender"""

    RECOMMENDER_NAME = "ImplicitALSRecommender"

    def __init__(self, URM_train, verbose=True):
        super().__init__(URM_train, verbose)

    def fit(self, epochs = 300, num_factors=50, regularization=0.01):
        self.model = implicit.als.AlternatingLeastSquares(factors=num_factors, regularization=regularization,
                                                          iterations=epochs, calculate_training_loss=True)
        self.model.fit(self.URM_train.T.tocsr())
        self.USER_factors = self.model.user_factors
        self.ITEM_factors = self.model.item_factors

