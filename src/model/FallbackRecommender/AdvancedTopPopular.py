from course_lib.Base.BaseRecommender import BaseRecommender
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import numpy as np
import pandas as pd

from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender


class AdvancedTopPopular(BaseRecommender):
    RECOMMENDER_NAME = "AdvancedTopPopular"

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
        self.clusters = None
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

    @staticmethod
    def _preprocess_data_frame_(df: pd.DataFrame, mapper_dict):
        """
        Pre-process the data frame removing the users discarded from the split in train-test.
        Then, change the index of the data frame (i.e. users number) according to mapper of the split.

        :param df: data frame read from file
        :param mapper_dict: dictionary mapping between original users, and then one in the split
        :return: preprocessed data frame
        """
        # Here, we have the original users, we should map them to the user in the split
        warm_user_original = df.index
        keys = np.array(list(mapper_dict.keys()))  # Users that are actually mapped to something else

        mask = np.in1d(warm_user_original, keys)
        not_mask = np.logical_not(mask)
        user_to_discard = warm_user_original[not_mask]  # users that should be removed from the data frame since they

        # Removing users discarded from the data frame
        new_df = df.drop(user_to_discard, inplace=False)

        # Mapping the users
        new_df = new_df.reset_index()
        new_df['user'] = new_df['user'].map(lambda x: mapper_dict[str(x)])
        new_df = new_df.set_index("user")

        return new_df

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
        # Verifying clustering method is present
        if self._verify_clustering_method_(clustering_method):
            raise RuntimeError("Clustering method should in in ['kmodes','kproto']")
        if self._verify_init_method_(init_method):
            raise RuntimeError("Clustering init method should be in ['Huang', 'Cao', 'random']")

        # Pre-processing the data frame
        self.data = self._preprocess_data_frame_(self.data, self.mapper_dict)

        # Clustering items
        if clustering_method == "kmodes":
            km = KModes(n_clusters=n_clusters, init='Huang', n_init=n_init, verbose=verbose, random_state=seed,
                        n_jobs=n_jobs)
        else:
            km = KPrototypes(n_clusters=n_clusters, n_init=n_init, verbose=verbose, random_state=seed, n_jobs=n_jobs)

        self.clusters = km.fit_predict(self.data)

        # Fitting top popular methods to the various clusters of users
        # Get users clustered
        for i in range(0, n_clusters):
            self.users_clustered_list.append(np.where(self.clusters == i))

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
        return self.hybrid_recommender.recommend(user_id_array, items_to_compute)

    def save_model(self, folder_path, file_name=None):
        raise NotImplemented()
