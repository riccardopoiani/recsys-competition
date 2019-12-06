from abc import ABC

from course_lib.Base.BaseRecommender import BaseRecommender
from scipy.sparse.csr import csr_matrix

from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender
from course_lib.Base.Recommender_utils import similarityMatrixTopK


class HybridMixedSimilarityRecommender(BaseRecommender, ABC):
    """
    Hybrid model that merges different similarity matrix of the same shape for doing computations
    """

    RECOMMENDER_NAME = "ItemMixedSimilarity"

    def __init__(self, URM_train):
        self.topK = 0
        self.alpha = 0
        self.W_sparse_list: list = []
        self.W_sparse = None
        super().__init__(URM_train)

    def _verify_sparse_matrix_(self, new_W_sparse: csr_matrix):
        if len(self.W_sparse_list) == 0:
            return True
        if self.W_sparse_list[0].shape != new_W_sparse.shape:
            return False
        return True

    def get_list_size(self):
        return len(self.W_sparse_list)

    def clear_sparse_matrix(self):
        """
        Reset the similarity matrix of the hybrid model
        :return: None
        """
        self.W_sparse = []

    def add_similarity_matrix(self, new_W_sparse):
        """
        Add a new similarity matrix.
        It has to be of the same shape of the ones already added

        :param new_W_sparse: matrix that will be added to the model
        :return: None
        """
        if len(self.W_sparse_list) == 0:
            self.W_sparse_list.append(new_W_sparse)
        else:
            if self._verify_sparse_matrix_(new_W_sparse):
                self.W_sparse_list.append(new_W_sparse)
            else:
                raise RuntimeError("Matrix dimension should be {}, but {} was found".format(self.W_sparse[0].shape,
                                                                                            new_W_sparse.shape))

    def copy(self):
        raise NotImplemented()


class ItemHybridModelRecommender(HybridMixedSimilarityRecommender, BaseItemSimilarityMatrixRecommender):
    """
    Item similarities matrices are used in this case
    """

    def __init__(self, URM_train):
        super().__init__(URM_train)

    def fit(self, topK=100, alpha1=1, alpha2=1, alpha3=1):
        self.topK = topK
        alpha_list = [alpha1, alpha2, alpha3]

        if len(alpha_list) != len(self.W_sparse_list):
            raise RuntimeError("Weighting list is not right. {} expected, {} found".format(len(self.W_sparse_list),
                                                                                           len(alpha_list)))
        self.W_sparse = alpha_list[0] * self.W_sparse_list[0]

        for i in range(1, len(alpha_list)):
            self.W_sparse += alpha_list[i] * self.W_sparse_list[i]

        self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK).tocsr()

    def copy(self):
        copy = ItemHybridModelRecommender(URM_train=self.URM_train)
        copy.topK = 0
        copy.W_sparse_list = self.W_sparse_list
        copy.W_sparse = None
        return copy


class UserHybridModelRecommender(HybridMixedSimilarityRecommender, BaseUserSimilarityMatrixRecommender):
    def __init__(self, URM_train):
        super().__init__(URM_train=URM_train)

    def fit(self, topK=100, alpha1=1, alpha2=1, alpha3=1):
        self.topK = topK
        alpha_list = [alpha1, alpha2, alpha3]
        if len(alpha_list) != len(self.W_sparse_list):
            raise RuntimeError("Weighting list is not right. {} expected, {} found".format(len(self.W_sparse_list),
                                                                                           len(alpha_list)))

        self.W_sparse = alpha_list[0] * self.W_sparse_list[0]

        for i in range(1, len(alpha_list)):
            self.W_sparse += alpha_list[i] * self.W_sparse_list[i]

        self.W_sparse = similarityMatrixTopK(self.W_sparse, k=self.topK).tocsr()

    def copy(self):
        copy = UserHybridModelRecommender(URM_train=self.URM_train)
        copy.topK = 0
        copy.W_sparse_list = self.W_sparse_list
        copy.W_sparse = None
        return copy
