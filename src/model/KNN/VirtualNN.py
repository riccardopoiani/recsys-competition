from abc import ABC

import numpy as np
from KNN.UserKNNCFRecommender import UserKNNCFRecommender
from scipy.sparse import csr_matrix, vstack, triu

from course_lib.Base.BaseRecommender import BaseRecommender
from course_lib.Base.BaseSimilarityMatrixRecommender import BaseItemSimilarityMatrixRecommender, \
    BaseUserSimilarityMatrixRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender


class VirtualNearestNeighbor(BaseRecommender, ABC):
    """
    From
    "Improving Collaborative Filtering’s Rating Prediction Coverage in Sparse Datasets
    through the Introduction of Virtual Near Neighbors" - Margaris et. al
    """

    def __init__(self, URM_train):
        super().__init__(URM_train)

    @classmethod
    def newAugmentUMR(cls, URM_train: csr_matrix, W_sparse: csr_matrix,
                      threshold_interactions: int, threshold_similarity: float):
        print("New Augmenting URM")
        count_W_sparse = URM_train.dot(URM_train.transpose())
        count_mask: csr_matrix = count_W_sparse > threshold_interactions
        sim_mask: csr_matrix = W_sparse > threshold_similarity
        mask = count_mask.multiply(sim_mask)
        mask = triu(mask)
        mask = mask.tocoo()

        row_user = mask.row
        col_user = mask.col

        new_mask = row_user != col_user
        row_user = row_user[new_mask]
        col_user = col_user[new_mask]
        new_users = np.array([row_user, col_user])
        new_users = np.transpose(new_users)
        new_rows_list: list = new_users.tolist()

        print("Candidate list size: {}".format(len(new_rows_list)))

        # Creating the new matrix
        print("Creating new URM...", end="")
        new_URM = None
        for candidate in new_rows_list:
            new_row = URM_train[[candidate[0], candidate[1]]].sum(axis=0)
            new_row = csr_matrix(new_row)
            new_row.data[new_row.data > 1] = 1

            if new_URM is None:
                new_URM = new_row
            else:
                new_URM = vstack([new_URM, new_row], format="csr")

        if new_URM is None:
            new_URM = URM_train
        else:
            new_URM = vstack([URM_train, new_URM], format="csr")

        print("Done")

        return new_URM

    @classmethod
    def augmentURM(cls, URM_train: csr_matrix, W_sparse: csr_matrix,
                   threshold_interactions: int, threshold_similarity: float):
        """
        Augmentation of the URM train.

        :param threshold_interactions: here a threshold on the similarity is considered.
        Similarity matrix W_sparse will be considered for this purpose
        :param threshold_similarity: threshold used to insert a new row.
        In this case it is specified as the minimum number of interactions required to insert a new
        row in the URM train
        :param W_sparse: similarity matrix
        :param URM_train: URM train that will be augmented
        :return: a csr_matrix with augmented interactions according to the threshold
        """
        print("Augmenting URM")
        URM_train = URM_train.copy()

        # Count similarity
        count_W_sparse = URM_train.dot(URM_train.transpose())

        # Selecting new
        print("Selecting new candidates")
        users = np.arange(URM_train.shape[0])
        new_rows_list = []
        for i in range(0, users.size):
            if i % 5000 == 0:
                print("{} done in {}".format(i, users.size))
            candidates = count_W_sparse[i].indices  # users candidates
            data = count_W_sparse[i].data  # data for the candidates

            for j, candidate in enumerate(candidates):
                if candidate > i and data[j] > threshold_interactions and W_sparse[i, candidate] > threshold_similarity:
                    new_rows_list.append([i, candidate])

        print("Candidate list size: {}".format(len(new_rows_list)))

        # Creating the new matrix
        print("Creating new URM...", end="")
        new_URM = None
        for candidate in new_rows_list:
            new_row = URM_train[[candidate[0], candidate[1]]].sum(axis=0)
            new_row = csr_matrix(new_row)
            new_row.data[new_row.data > 1] = 1

            if new_URM is None:
                new_URM = new_row
            else:
                new_URM = vstack([new_URM, new_row], format="csr")

        if new_URM is None:
            new_URM = URM_train
        else:
            new_URM = vstack([URM_train, new_URM], format="csr")

        print("Done")

        return new_URM


class ItemVNN(VirtualNearestNeighbor, BaseItemSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "ItemVNN"

    def __init__(self, URM_train):
        self.W_sparse = None
        super().__init__(URM_train)

    def fit(self, threshold_sim=0.5, threshold_count=5, topK=50, shrink=100, similarity='cosine', normalize=True,
            feature_weighting="none",
            **similarity_args):
        user_cf = UserKNNCFRecommender(self.URM_train)
        user_cf.fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize,
                    feature_weighting=feature_weighting,
                    **similarity_args)
        new_URM = VirtualNearestNeighbor.newAugmentUMR(URM_train=self.URM_train, W_sparse=user_cf.W_sparse,
                                                    threshold_interactions=threshold_count,
                                                    threshold_similarity=threshold_sim)

        new_item_cf = ItemKNNCFRecommender(new_URM)
        new_item_cf.fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize,
                        feature_weighting=feature_weighting, **similarity_args)
        self.W_sparse = new_item_cf.W_sparse


class UserVNN(VirtualNearestNeighbor, BaseUserSimilarityMatrixRecommender):
    RECOMMENDER_NAME = "UserVNN"

    def __init__(self, URM_train):
        self.W_sparse = None
        super().__init__(URM_train=URM_train)

    def fit(self, threshold_sim=0.5, threshold_count=5, topK=50, shrink=100, similarity='cosine', normalize=True,
            feature_weighting="none", **similarity_args):
        # Augmenting URM
        user_cf = UserKNNCFRecommender(self.URM_train)
        user_cf.fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize,
                    feature_weighting=feature_weighting,
                    **similarity_args)
        new_URM = VirtualNearestNeighbor.newAugmentUMR(URM_train=self.URM_train, W_sparse=user_cf.W_sparse,
                                                    threshold_interactions=threshold_count,
                                                    threshold_similarity=threshold_sim)

        new_user_cf = UserKNNCFRecommender(new_URM)
        new_user_cf.fit(topK=topK, shrink=shrink, similarity=similarity, normalize=normalize,
                        feature_weighting=feature_weighting,
                        **similarity_args)
        self.W_sparse = new_user_cf.W_sparse
        self.URM_train = new_URM
