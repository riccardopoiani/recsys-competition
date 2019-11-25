from course_lib.Base.BaseRecommender import BaseRecommender


class AdvancedTopPopular(BaseRecommender):


    def __init__(self, URM_train, ICM, UCM):
        self.UCM = UCM.copy()
        self.ICM = ICM.copy()
        super().__init__(URM_train)

    def fit(self):
        # Compute similarity matrix to be used for
        pass

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        pass
