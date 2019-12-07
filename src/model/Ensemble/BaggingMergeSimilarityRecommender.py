from abc import ABC

from tqdm import tqdm

from course_lib.Base.BaseMatrixFactorizationRecommender import BaseMatrixFactorizationRecommender
from course_lib.Base.BaseRecommender import BaseRecommender
from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender
from course_lib.Base.Recommender_utils import similarityMatrixTopK
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython
from src.model.Ensemble.BaggingUtils import get_bootstrap_URM
from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender
from src.utils.general_utility_functions import block_print, enable_print


class BaggingMergeRecommender(BaseRecommender, ABC):

    COMPATIBLE_RECOMMENDERS = []

    def __init__(self, URM_train, recommender_class, **recommender_constr_kwargs):
        if not (recommender_class in self.COMPATIBLE_RECOMMENDERS):
            raise ValueError("The only compatible recommenders are: {}".format(self.COMPATIBLE_RECOMMENDERS))

        super().__init__(URM_train)
        self.recommender_class = recommender_class
        self.recommender_kwargs = recommender_constr_kwargs
        self.models = []

    def fit(self, num_models=5, hyper_parameters_range=None, **kwargs):
        """
        Fit all the num_models and merge them into a unique model

        :param topK: if less than 0, KNN is not applied on the similarity
        :param num_models: number of models in the bagging recommender
        :param hyper_parameters_range: hyper parameters range to give some more diversity in models
        """
        if hyper_parameters_range is None:
            hyper_parameters_range = {}

        for i in tqdm(range(num_models), desc="Fitting bagging models"):
            URM_bootstrap = get_bootstrap_URM(self.URM_train)
            parameters = {}
            for parameter_name, parameter_range in hyper_parameters_range.items():
                parameters[parameter_name] = parameter_range.rvs()

            block_print()
            recommender_object = self.recommender_class(URM_bootstrap, **self.recommender_kwargs)
            recommender_object.fit(**parameters)
            enable_print()

            self.models.append(recommender_object)


class BaggingMergeItemSimilarityRecommender(BaggingMergeRecommender, BaseItemSimilarityMatrixRecommender):
    """
    Bagging Merge Item Similarity Recommender: samples with replacement only the positive interactions and groups
    the model by merging them
    """

    RECOMMENDER_NAME = "BaggingMergeItemSimilarityRecommender"

    COMPATIBLE_RECOMMENDERS = [ItemKNNCBFRecommender, ItemKNNCFRecommender, SLIM_BPR_Cython]

    def __init__(self, URM_train, recommender_class, **recommender_constr_kwargs):
        super().__init__(URM_train, recommender_class, **recommender_constr_kwargs)
        self.W_sparse = None

    def fit(self, topK=-1, num_models=5, hyper_parameters_range=None):
        super().fit(num_models, hyper_parameters_range)

        # Build new similarity matrix
        self.W_sparse = self.models[0].W_sparse
        for i in range(1, len(self.models)):
            self.W_sparse = self.W_sparse + self.models[i].W_sparse
        self.W_sparse = self.W_sparse / len(self.models)

        if topK >= 0:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=topK).tocsr()


class BaggingMergeUserSimilarityRecommender(BaggingMergeRecommender, BaseUserSimilarityMatrixRecommender):
    """
    Bagging Merge User Similarity Recommender: samples with replacement only the positive interactions and
    groups the model by merging them
    """

    RECOMMENDER_NAME = "BaggingMergeUserSimilarityRecommender"

    COMPATIBLE_RECOMMENDERS = [UserKNNCBFRecommender, UserKNNCFRecommender]

    def __init__(self, URM_train, recommender_class, **recommender_constr_kwargs):
        super().__init__(URM_train, recommender_class, **recommender_constr_kwargs)
        self.W_sparse = None

    def fit(self, topK=-1, num_models=5, hyper_parameters_range=None):
        super().fit(num_models, hyper_parameters_range)

        # Build new similarity matrix
        self.W_sparse = self.models[0].W_sparse
        for i in range(1, len(self.models)):
            self.W_sparse = self.W_sparse + self.models[i].W_sparse
        self.W_sparse = self.W_sparse / len(self.models)

        if topK >= 0:
            self.W_sparse = similarityMatrixTopK(self.W_sparse, k=topK).tocsr()


class BaggingMergeFactorsRecommender(BaggingMergeRecommender, BaseMatrixFactorizationRecommender):

    RECOMMENDER_NAME = "BaggingMergeFactorsRecommender"

    COMPATIBLE_RECOMMENDERS = [ImplicitALSRecommender, LogisticMFRecommender, MF_BPR_Recommender]

    def __init__(self, URM_train, recommender_class, **recommender_constr_kwargs):
        super().__init__(URM_train, recommender_class, **recommender_constr_kwargs)
        self.USER_factors = None
        self.ITEM_factors = None

    def fit(self, num_models=5, hyper_parameters_range=None, **kwargs):
        super().fit(num_models, hyper_parameters_range, **kwargs)

        # Build new similarity matrix
        self.USER_factors = self.models[0].USER_factors
        self.ITEM_factors = self.models[0].ITEM_factors
        for i in range(1, len(self.models)):
            self.USER_factors = self.USER_factors + self.models[i].USER_factors
            self.ITEM_factors = self.ITEM_factors + self.models[i].ITEM_factors
        self.USER_factors = self.USER_factors / len(self.models)
        self.ITEM_factors = self.ITEM_factors / len(self.models)

