from abc import ABC

import numpy as np
import scipy.sparse as sps

from course_lib.Data_manager.DataReader import DataReader
from course_lib.Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs
from src.data_management.RecSys2019Reader_utils import build_ICM_all, build_UCM_all
from src.feature.feature_processing import transform_numerical_to_label


class AbstractDataPreprocessing(DataReader, ABC):
    DATASET_SUBFOLDER = "abstract_preprocessing/"

    def __init__(self, reader: DataReader, soft_copy=True):
        super().__init__()

        self.reader = reader
        self.soft_copy = soft_copy
        self._LOADED_UCM_DICT = {}
        self._LOADED_UCM_MAPPER_DICT = {}

    def _load_from_original_file(self):
        # Load from original files for data preprocessing is useless since there is no save point for this
        pass

    def _get_dataset_name_root(self):
        return self.reader._get_dataset_name_root() + self.DATASET_SUBFOLDER

    def _get_dataset_name(self):
        return self.reader._get_dataset_name() + self.DATASET_SUBFOLDER

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        raise NotImplementedError("_preprocess_URM_all is not implemented in the abstract class")

    def _preprocess_ICMs(self):
        raise NotImplementedError("_preprocess_ICMs is not implemented in the abstract class")

    def load_data(self, save_folder_path=None):
        self.reader.load_data()

        self._copy_static_data_reader()
        self._copy_data_reader()

        URM_all = self._LOADED_URM_DICT["URM_all"]

        self._LOADED_URM_DICT["URM_all"] = self._preprocess_URM_all(URM_all)
        self._preprocess_ICMs()

    def _copy_static_data_reader(self):
        self.IS_IMPLICIT = self.reader.IS_IMPLICIT
        self.AVAILABLE_ICM = self.reader.get_all_available_ICM_names()
        self.ICM_to_load_list = self.reader.get_loaded_ICM_names()
        self.AVAILABLE_URM = self.reader.get_loaded_URM_names()
        self.AVAILABLE_UCM = self.reader.get_loaded_UCM_names()  # ignore this error since there is for us

    def get_UCM_from_name(self, UCM_name):
        return self._LOADED_UCM_DICT[UCM_name]

    def get_loaded_UCM_dict(self):
        UCM_dict = {}

        for UCM_name in self.get_loaded_UCM_names():
            UCM_dict[UCM_name] = self.get_UCM_from_name(UCM_name)

        return UCM_dict

    def get_UCM_feature_to_index_mapper_from_name(self, UCM_name):
        self._assert_is_initialized()
        return self._LOADED_UCM_MAPPER_DICT[UCM_name].copy()

    def get_all_available_UCM_names(self):
        return self.AVAILABLE_UCM.copy()

    def get_loaded_UCM_names(self):
        return self.AVAILABLE_UCM.copy()

    def _copy_data_reader(self):
        if self.soft_copy:
            self._LOADED_URM_DICT = self.reader._LOADED_URM_DICT
            self._LOADED_ICM_DICT = self.reader._LOADED_ICM_DICT
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader._LOADED_GLOBAL_MAPPER_DICT
            self._LOADED_ICM_MAPPER_DICT = self.reader._LOADED_ICM_MAPPER_DICT
            self._LOADED_UCM_DICT = self.reader._LOADED_UCM_DICT  # ignore this error since there is for us
            self._LOADED_UCM_MAPPER_DICT = self.reader._LOADED_UCM_MAPPER_DICT  # ignore this error since there is for us
        else:
            self._LOADED_URM_DICT = self.reader.get_loaded_URM_dict()
            self._LOADED_ICM_DICT = self.reader.get_loaded_ICM_dict()
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader.get_loaded_Global_mappers()
            self._LOADED_ICM_MAPPER_DICT = self.reader._LOADED_ICM_MAPPER_DICT
            self._LOADED_UCM_DICT = self.reader.get_loaded_UCM_dict()  # ignore this error since there is for us
            self._LOADED_UCM_MAPPER_DICT = self.reader._LOADED_UCM_MAPPER_DICT  # ignore this error since there is for us
            print("Data Preprocessing: WARNING - ICM MAPPER DICT deep copy is not implemented")
            print("Data Preprocessing: WARNING - UCM MAPPER DICT deep copy is not implemented")


class DataPreprocessingRemoveColdUsersItems(AbstractDataPreprocessing):
    DATASET_SUBFOLDER = "cold/"

    def __init__(self, reader: DataReader, threshold_items=0, threshold_users=0, soft_copy=True):
        super().__init__(reader, soft_copy)
        self.threshold_items = threshold_items
        self.threshold_users = threshold_users

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > self.threshold_items
        self.warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

        URM_all = URM_all[:, self.warm_items]

        warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > self.threshold_users
        self.warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

        URM_all = URM_all[self.warm_users, :]

        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(
            self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"],
            np.arange(0, len(warm_users_mask), dtype=np.int)[np.logical_not(warm_users_mask)])

        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = reconcile_mapper_with_removed_tokens(
            self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"],
            np.arange(0, len(warm_items_mask), dtype=np.int)[np.logical_not(warm_items_mask)])

        return URM_all

    def _preprocess_ICMs(self):
        for ICM_name, ICM_object in self._LOADED_ICM_DICT.items():
            self._LOADED_ICM_DICT[ICM_name] = ICM_object[self.warm_items, :]
