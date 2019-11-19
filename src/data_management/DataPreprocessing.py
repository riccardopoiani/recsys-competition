
from course_lib.Data_manager.DataReader import DataReader
import scipy.sparse as sps
import numpy as np

class AbstractDataPreprocessing(DataReader):

    def __init__(self, reader: DataReader, soft_copy = True):
        super().__init__()
        self.reader = reader
        self.soft_copy = soft_copy

        self._copy_static_data_reader(reader)

    def _preprocess_URM_all(self, URM_all : sps.csr_matrix):
        raise NotImplementedError("_preprocess_URM_all was not implemented in the abstract class")

    def _preprocess_ICMs(self, reader : DataReader):
        raise NotImplementedError("_preprocess_ICM_all was not implemented in the abstract class")

    def load_data(self, save_folder_path = None):
        self.reader.load_data()

        self._copy_data_reader(self.reader)

        URM_all = self._LOADED_URM_DICT["URM_all"]

        self._LOADED_URM_DICT["URM_all"] = self._preprocess_URM_all(URM_all)
        self._LOADED_ICM_DICT = self._preprocess_ICMs(self.reader)

    def _copy_static_data_reader(self, reader):
        self.IS_IMPLICIT = reader.IS_IMPLICIT
        self.AVAILABLE_ICM = reader.get_all_available_ICM_names()
        self.ICM_to_load_list = reader.get_loaded_ICM_names()
        self.AVAILABLE_URM = reader.get_loaded_URM_names()

    def _copy_data_reader(self, reader):
        if self.soft_copy:
            self._LOADED_URM_DICT = self.reader._LOADED_URM_DICT
            self._LOADED_ICM_DICT = reader._LOADED_ICM_DICT
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader._LOADED_GLOBAL_MAPPER_DICT
            self._LOADED_ICM_MAPPER_DICT = reader._LOADED_ICM_MAPPER_DICT
        else:
            self._LOADED_URM_DICT = self.reader.get_loaded_URM_dict()
            self._LOADED_ICM_DICT = reader.get_loaded_ICM_dict()
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader.get_loaded_Global_mappers()
            self._LOADED_ICM_MAPPER_DICT = reader._LOADED_ICM_MAPPER_DICT


class DataPreprocessingExample(AbstractDataPreprocessing):

    DATASET_SUBFOLDER = "preprocess_example/"

    def __init__(self, reader: DataReader, multiplier: int = 2, soft_copy = True):
        super().__init__(reader, soft_copy)
        self.multiplier = multiplier

    def _get_dataset_name_root(self):
        return self.reader._get_dataset_name_root() + self.DATASET_SUBFOLDER

    def _get_dataset_name(self):
        return self.reader._get_dataset_name() + self.DATASET_SUBFOLDER

    def _preprocess_URM_all(self, URM_all : sps.csr_matrix):
        URM_all.data = np.multiply(URM_all.data, self.multiplier)
        return URM_all

    def _preprocess_ICMs(self, reader : DataReader):
        return self._LOADED_ICM_DICT

class DataPreprocessingRemoveColdUsersItems(AbstractDataPreprocessing):

    DATASET_SUBFOLDER = "removed_cold_users/"

    def __init__(self, reader: DataReader, threshold_items=0, threshold_users=0, soft_copy=True):
        super().__init__(reader, soft_copy)
        self.threshold_items = threshold_items
        self.threshold_users = threshold_users


    def _get_dataset_name_root(self):
        return self.reader._get_dataset_name_root() + self.DATASET_SUBFOLDER

    def _get_dataset_name(self):
        return self.reader._get_dataset_name() + self.DATASET_SUBFOLDER

    def _preprocess_URM_all(self, URM_all : sps.csr_matrix):
        warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > self.threshold_items
        self.warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

        URM_all = URM_all[:, self.warm_items]

        warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > self.threshold_users
        warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

        URM_all = URM_all[warm_users, :]

        user_mapper = self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"]
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = {key: value for key, value in user_mapper.items()
                                                                        if value in warm_users}

        item_mapper = self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"]
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = {key: value for key, value in item_mapper.items()
                                                                        if value in self.warm_items}

        return URM_all

    def _preprocess_ICMs(self, reader : DataReader):
        for ICM_name, ICM_object in self._LOADED_ICM_DICT.items():
            self._LOADED_ICM_DICT[ICM_name] = ICM_object[self.warm_items, :]
        return self._LOADED_ICM_DICT
