from course_lib.Base.BaseRecommender import BaseRecommender
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import numpy as np

from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
from src.feature.clustering_utils import preprocess_data_frame, cluster_data


class AdvancedTopPopular(BaseRecommender):
    RECOMMENDER_NAME = "AdvancedTopPop"

    """
    Advanced top popular recommender, based on clustering.
    This method basically clusters the users according to some criteria, and, then, build a top popular
    recommender among clusters.
    """

    def __init__(self, URM_train, data, mapper_dict):
        """
        :param URM_train: extended from super-class
        :param data: dataframe containing information from the UCM about the users
        """
        self.mapper_dict = mapper_dict
        self.data = data
        self.users_clustered_list = []
        self.hybrid_recommender = HybridDemographicRecommender(URM_train)
        super().__init__(URM_train)

    @staticmethod
    def _verify_clustering_method_(clustering_method):
        if clustering_method not in ['kmodes', 'kproto']:
            return False
        return True

    @staticmethod
    def _verify_init_method_(init):
        if init not in ["Huang", "random", "Cao"]:
            return False
        return True

    @staticmethod
    def _build_mapper_(new_users: np.array):
        new_mapper = {}
        for i in range(0, new_users.size):
            new_mapper[str(new_users[i])] = i
        return new_mapper

    def fit(self, n_clusters=5, n_init=5, clustering_method="kmodes", verbose=1,
            seed=69420, init_method="Huang", n_jobs=1):
        """
        Creates clusters based on the data frame given in input.
        After that, it fits top popular method using the clusters, and, predicts accordingly to them.

        :param n_clusters: number of clusters
        :param n_init: number of initialization for the clustering method
        :param clustering_method: clustering method to be used: KModes and KProtoypes are available
        :param verbose: if you want to have verbose output 1, else 0
        :param seed: seed for the clustering randomness
        :param init_method: initialization method for the clusters
        :param n_jobs: number of jobs
        :return: user clustered
        """
        self.hybrid_recommender.reset_groups()

        # Verifying clustering method is present
        if not self._verify_clustering_method_(clustering_method):
            raise RuntimeError("Clustering method should in in ['kmodes','kproto']")
        if not self._verify_init_method_(init_method):
            raise RuntimeError("Clustering init method should be in ['Huang', 'Cao', 'random']")

        # Pre-processing the data frame
        self.data = preprocess_data_frame(self.data, self.mapper_dict)

        # Clustering
        self.users_clustered_list = cluster_data(self.data, clustering_method=clustering_method, n_jobs=n_jobs,
                                                 n_clusters=n_clusters, n_init=n_init, seed=seed,
                                                 init_method=init_method)

        # Fitting top populars and add them to the demographic hybrid
        for i in range(0, n_clusters):
            URM_train_cluster = self.URM_train[self.users_clustered_list[i]]
            top_pop = TopPop(URM_train_cluster)
            top_pop.fit()
            self.hybrid_recommender.add_user_group(i, self.users_clustered_list[i])
            self.hybrid_recommender.add_relation_recommender_group(top_pop, i)

        self.hybrid_recommender.fit()

        return self.users_clustered_list

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        return self.hybrid_recommender._compute_item_score(user_id_array, items_to_compute)

    def save_model(self, folder_path, file_name=None):
        raise NotImplemented()
