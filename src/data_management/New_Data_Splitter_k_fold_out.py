import pickle
from course_lib.Base.DataIO import DataIO
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_k_fold
from course_lib.Data_manager.DataReader_utils import *


class New_DataSplitter_Warm_k_fold(DataSplitter_k_fold):
    """
    This splitter performs a Holdout from the full URM splitting in train, test and validation
    Ensures that every user has at least an interaction in all splits
    """
    DATA_SPLITTER_NAME = "New_DataSplitter_Warm_k_fold"

    SPLIT_URM_DICT = None
    SPLIT_ICM_DICT = None
    SPLIT_ICM_MAPPER_DICT = None
    SPLIT_GLOBAL_MAPPER_DICT = None

    def __init__(self, dataReader_object, n_folds=5, forbid_new_split=False,
                 allow_cold_users=False):

        self.allow_cold_users = allow_cold_users

        super(New_DataSplitter_Warm_k_fold, self).__init__(dataReader_object,
                                                           n_folds=n_folds,
                                                           forbid_new_split=forbid_new_split)

    def _get_split_subfolder_name(self):
        """

        :return: warm_{n_folds}_fold/
        """
        return "warm_{}_fold/".format(self.n_folds)

    def _split_data_from_original_dataset(self, save_folder_path):

        self.dataReader_object.load_data()
        URM = self.dataReader_object.get_URM_all()

        # Managing data reader
        self.SPLIT_GLOBAL_MAPPER_DICT = {}
        for mapper_name, mapper_object in self.dataReader_object.get_loaded_Global_mappers().items():
            self.SPLIT_GLOBAL_MAPPER_DICT[mapper_name] = mapper_object.copy()

        URM = sps.csr_matrix(URM)

        if not self.allow_cold_users:
            user_interactions = np.ediff1d(URM.indptr)
            user_to_preserve = user_interactions >= self.n_folds
            user_to_remove = np.logical_not(user_to_preserve)

            print(
                "DataSplitter_Warm: Removing {} of {} users because they have less interactions than the number of folds".format(
                    URM.shape[0] - user_to_preserve.sum(), URM.shape[0]))

            URM = URM[user_to_preserve, :]

            self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(
                self.SPLIT_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
                np.arange(0, len(user_to_remove), dtype=np.int)[user_to_remove])

        self.n_users, self.n_items = URM.shape

        URM = sps.csr_matrix(URM)

        # Create empty URM for each fold
        self.fold_split = {}

        for fold_index in range(self.n_folds):
            self.fold_split[fold_index] = {}
            self.fold_split[fold_index]["URM"] = sps.coo_matrix(URM.shape)

            URM_fold_object = self.fold_split[fold_index]["URM"]
            # List.extend is waaaay faster than numpy.concatenate
            URM_fold_object.row = []
            URM_fold_object.col = []
            URM_fold_object.data = []

        for user_id in range(self.n_users):

            start_user_position = URM.indptr[user_id]
            end_user_position = URM.indptr[user_id + 1]

            user_profile = URM.indices[start_user_position:end_user_position]

            indices_to_suffle = np.arange(len(user_profile), dtype=np.int)

            np.random.shuffle(indices_to_suffle)

            user_profile = user_profile[indices_to_suffle]
            user_interactions = URM.data[start_user_position:end_user_position][indices_to_suffle]

            # interactions_per_fold is a float number, to auto-adjust fold size
            interactions_per_fold = len(user_profile) / self.n_folds

            for fold_index in range(self.n_folds):

                start_pos = int(interactions_per_fold * fold_index)
                end_pos = int(interactions_per_fold * (fold_index + 1))

                if fold_index == self.n_folds - 1:
                    end_pos = len(user_profile)

                current_fold_user_profile = user_profile[start_pos:end_pos]
                current_fold_user_interactions = user_interactions[start_pos:end_pos]

                URM_fold_object = self.fold_split[fold_index]["URM"]

                URM_fold_object.row.extend([user_id] * len(current_fold_user_profile))
                URM_fold_object.col.extend(current_fold_user_profile)
                URM_fold_object.data.extend(current_fold_user_interactions)

        for fold_index in range(self.n_folds):
            URM_fold_object = self.fold_split[fold_index]["URM"]
            URM_fold_object.row = np.array(URM_fold_object.row, dtype=np.int)
            URM_fold_object.col = np.array(URM_fold_object.col, dtype=np.int)
            URM_fold_object.data = np.array(URM_fold_object.data, dtype=np.float)

            self.fold_split[fold_index]["URM"] = sps.csr_matrix(URM_fold_object)
            self.fold_split[fold_index]["items_in_fold"] = np.arange(0, self.n_items, dtype=np.int)

        fold_dict_to_save = {"fold_split": self.fold_split,
                             "n_folds": self.n_folds,
                             "n_items": self.n_items,
                             "n_users": self.n_users,
                             "allow_cold_users": self.allow_cold_users,
                             }

        if self.allow_cold_users:
            allow_user = "allow_cold_users"
        else:
            allow_user = "only_warm_users"

        pickle.dump(fold_dict_to_save,
                    open(save_folder_path + "URM_{}_fold_split_{}".format(self.n_folds, allow_user), "wb"),
                    protocol=pickle.HIGHEST_PROTOCOL)

        for ICM_name in self.dataReader_object.get_loaded_ICM_names():
            pickle.dump(self.dataReader_object.get_ICM_from_name(ICM_name),
                        open(save_folder_path + "{}".format(ICM_name), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(self.dataReader_object.get_ICM_feature_to_index_mapper_from_name(ICM_name),
                        open(save_folder_path + "tokenToFeatureMapper_{}".format(ICM_name), "wb"),
                        protocol=pickle.HIGHEST_PROTOCOL)

            # MAPPER MANAGING
            if self.allow_cold_users:
                allow_cold_users_suffix = "allow_cold_users"
            else:
                allow_cold_users_suffix = "only_warm_users"
            name_suffix = "_{}".format(allow_cold_users_suffix)
            dataIO = DataIO(folder_path=save_folder_path)
            dataIO.save_data(data_dict_to_save=self.SPLIT_GLOBAL_MAPPER_DICT,
                             file_name="split_mappers" + name_suffix)


        print("DataSplitter: Split complete")

    def _load_previously_built_split_and_attributes(self, save_folder_path):
        """
        Loads all URM and ICM
        :return:
        """

        if self.allow_cold_users:
            allow_cold_users_file_name = "allow_cold_users"
        else:
            allow_cold_users_file_name = "only_warm_users"

        data_dict = pickle.load(
            open(save_folder_path + "URM_{}_fold_split_{}".format(self.n_folds, allow_cold_users_file_name), "rb"))

        for attrib_name in data_dict.keys():
            self.__setattr__(attrib_name, data_dict[attrib_name])

        for ICM_name in self.dataReader_object.get_loaded_ICM_names():
            ICM_object = pickle.load(open(save_folder_path + "{}".format(ICM_name), "rb"))
            self.__setattr__(ICM_name, ICM_object)

            pickle.load(open(save_folder_path + "tokenToFeatureMapper_{}".format(ICM_name), "rb"))
            self.__setattr__("tokenToFeatureMapper_{}".format(ICM_name), ICM_object)

        # MAPPER MANAGING
        if self.allow_cold_users:
            allow_cold_users_suffix = "allow_cold_users"
        else:
            allow_cold_users_suffix = "only_warm_users"
        name_suffix = "_{}".format(allow_cold_users_suffix)
        dataIO = DataIO(folder_path=save_folder_path)
        self.SPLIT_GLOBAL_MAPPER_DICT = dataIO.load_data(file_name="split_mappers" + name_suffix)

