from course_lib.Data_manager.DataReader import DataReader
import scipy.sparse as sps
import numpy as np

from course_lib.Data_manager.DataReader_utils import reconcile_mapper_with_removed_tokens, merge_ICM
from src.data_management.RecSys2019Reader_utils import build_ICM_all
from src.feature.feature_processing import transform_numerical_to_label
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs, IncrementalSparseMatrix


class AbstractDataPreprocessing(DataReader):

    def __init__(self, reader: DataReader, soft_copy=True):
        super().__init__()
        self.reader = reader
        self.soft_copy = soft_copy

        self._copy_static_data_reader()

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        raise NotImplementedError("_preprocess_URM_all is not implemented in the abstract class")

    def _preprocess_ICMs(self):
        raise NotImplementedError("_preprocess_ICMs is not implemented in the abstract class")

    def load_data(self, save_folder_path=None):
        self.reader.load_data()

        self._copy_data_reader()

        URM_all = self._LOADED_URM_DICT["URM_all"]

        self._LOADED_URM_DICT["URM_all"] = self._preprocess_URM_all(URM_all)
        self._LOADED_ICM_DICT = self._preprocess_ICMs()

    def _copy_static_data_reader(self):
        self.IS_IMPLICIT = self.reader.IS_IMPLICIT
        self.AVAILABLE_ICM = self.reader.get_all_available_ICM_names()
        self.ICM_to_load_list = self.reader.get_loaded_ICM_names()
        self.AVAILABLE_URM = self.reader.get_loaded_URM_names()

    def _copy_data_reader(self):
        if self.soft_copy:
            self._LOADED_URM_DICT = self.reader._LOADED_URM_DICT
            self._LOADED_ICM_DICT = self.reader._LOADED_ICM_DICT
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader._LOADED_GLOBAL_MAPPER_DICT
            self._LOADED_ICM_MAPPER_DICT = self.reader._LOADED_ICM_MAPPER_DICT
        else:
            self._LOADED_URM_DICT = self.reader.get_loaded_URM_dict()
            self._LOADED_ICM_DICT = self.reader.get_loaded_ICM_dict()
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader.get_loaded_Global_mappers()
            self._LOADED_ICM_MAPPER_DICT = self.reader._LOADED_ICM_MAPPER_DICT
            print("Data Preprocessing: WARNING - ICM MAPPER DICT deep copy is not implemented")


class DataPreprocessingExample(AbstractDataPreprocessing):
    DATASET_SUBFOLDER = "preprocess_example/"

    def __init__(self, reader: DataReader, multiplier: int = 2, soft_copy=True):
        super().__init__(reader, soft_copy)
        self.multiplier = multiplier

    def _get_dataset_name_root(self):
        return self.reader._get_dataset_name_root() + self.DATASET_SUBFOLDER

    def _get_dataset_name(self):
        return self.reader._get_dataset_name() + self.DATASET_SUBFOLDER

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        URM_all.data = np.multiply(URM_all.data, self.multiplier)
        return URM_all

    def _preprocess_ICMs(self):
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

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        warm_items_mask = np.ediff1d(URM_all.tocsc().indptr) > self.threshold_items
        self.warm_items = np.arange(URM_all.shape[1])[warm_items_mask]

        URM_all = URM_all[:, self.warm_items]

        warm_users_mask = np.ediff1d(URM_all.tocsr().indptr) > self.threshold_users
        warm_users = np.arange(URM_all.shape[0])[warm_users_mask]

        URM_all = URM_all[warm_users, :]

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
        return self._LOADED_ICM_DICT


class DataPreprocessingDigitizeICMs(AbstractDataPreprocessing):
    DATASET_SUBFOLDER = "preprocess_example/"

    def __init__(self, reader: DataReader, ICM_name_to_bins_mapper: dict, soft_copy=True):
        super().__init__(reader, soft_copy)
        self.ICM_name_to_bins_mapper = ICM_name_to_bins_mapper

    def _get_dataset_name_root(self):
        return self.reader._get_dataset_name_root() + self.DATASET_SUBFOLDER

    def _get_dataset_name(self):
        return self.reader._get_dataset_name() + self.DATASET_SUBFOLDER

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        return URM_all

    def _preprocess_ICMs(self):
        """
        Digitize all ICMs whose name is inside the ICM_name_to_bins_mapper and re-merge ICMs in order to obtain
        ICM_all

        :return:
        """
        if ~np.all(np.in1d(list(self.ICM_name_to_bins_mapper.keys()), list(self._LOADED_ICM_DICT.keys()))):
            raise KeyError("ICM_name_to_bins_mapper does contains wrong ICM names")

        # Digitize unskewed data of ICMs
        for ICM_name, bins in self.ICM_name_to_bins_mapper.items():
            ICM_object: sps.csr_matrix = self._LOADED_ICM_DICT[ICM_name]
            if ICM_object.shape[1] != 1:
                raise KeyError("ICM name passed is not regarding a single feature, thus, it cannot be digitized")

            x = np.array(ICM_object.data)
            #unskewed_x = np.log1p(1 / x)
            labelled_x = transform_numerical_to_label(x, bins)
            vectorized_change_label = np.vectorize(lambda elem: "%s-%d" % (ICM_name, elem))
            labelled_x = vectorized_change_label(labelled_x)

            item_original_ID_to_index_mapper = self._LOADED_GLOBAL_MAPPER_DICT['item_original_ID_to_index']
            ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index_mapper,
                                                            on_new_row="ignore")
            ICM_builder.add_data_lists(np.array(ICM_object.tocoo().row, dtype=str), labelled_x, np.ones(len(labelled_x), dtype=np.float32))
            self._LOADED_ICM_DICT[ICM_name] = ICM_builder.get_SparseMatrix()
            self._LOADED_ICM_MAPPER_DICT[ICM_name] = ICM_builder.get_column_token_to_id_mapper()

        # Re-build ICM_all
        if "ICM_all" in self.get_loaded_ICM_names():
            self._LOADED_ICM_DICT["ICM_all"], self._LOADED_ICM_MAPPER_DICT["ICM_all"] = build_ICM_all(self._LOADED_ICM_DICT,
                                                                                                      self._LOADED_ICM_MAPPER_DICT)

        return self._LOADED_ICM_DICT
