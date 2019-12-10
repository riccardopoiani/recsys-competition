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
            self._LOADED_UCM_MAPPER_DICT = self.reader._LOADED_UCM_MAPPER_DICT # ignore this error since there is for us
        else:
            self._LOADED_URM_DICT = self.reader.get_loaded_URM_dict()
            self._LOADED_ICM_DICT = self.reader.get_loaded_ICM_dict()
            self._LOADED_GLOBAL_MAPPER_DICT = self.reader.get_loaded_Global_mappers()
            self._LOADED_ICM_MAPPER_DICT = self.reader._LOADED_ICM_MAPPER_DICT
            self._LOADED_UCM_DICT = self.reader.get_loaded_UCM_dict()  # ignore this error since there is for us
            self._LOADED_UCM_MAPPER_DICT = self.reader._LOADED_UCM_MAPPER_DICT # ignore this error since there is for us
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


class DataPreprocessingFeatureEngineering(AbstractDataPreprocessing):
    DATASET_SUBFOLDER = "n/"

    def __init__(self, reader: DataReader, ICM_names_to_count: list, UCM_names_to_ICM=None, soft_copy=True):
        """
        Preprocessing operation to add new ICMs and UCMs.
         - The new UCM names from ICMs are called UCM_{}.format(ICM_suffix_name). For example, for ICM_age, the name of the new UCM is UCM_age
         - The new ICM names from UCMs applies the same principle of the above one

        :param reader: data reader
        :param ICM_names_to_count: list of ICM name to apply the preprocessing of count values
        :param ICM_names_to_UCM: list of ICM name to apply a dot product with URM in order to obtain new features for UCM
        :param UCM_names_to_ICM: list of UCM name to apply a dot product with URM in order to obtain new features for ICM
        :param soft_copy:
        """
        super().__init__(reader, soft_copy)
        if UCM_names_to_ICM is None:
            UCM_names_to_ICM = []

        self.ICM_names_to_count = ICM_names_to_count
        self.UCM_names_to_ICM = UCM_names_to_ICM

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        return URM_all

    def _preprocess_ICMs(self):
        if ~np.all(np.in1d(list(self.ICM_names_to_count), list(self._LOADED_ICM_DICT.keys()))):
            raise KeyError("Mapper contains wrong ICM names")

        if ~np.all(np.in1d(list(self.UCM_names_to_ICM), list(self._LOADED_UCM_DICT.keys()))):
            raise KeyError("Mapper contains wrong ICM names")

        for ICM_name in self.ICM_names_to_count:
            ICM_object: sps.csr_matrix = self._LOADED_ICM_DICT[ICM_name]
            column = ICM_object.tocoo().col
            uniques, counts = np.unique(column, return_counts=True)

            new_ICM_name = "{}_count".format(ICM_name)
            new_row = np.array(ICM_object.tocoo().row, dtype=str)
            new_col = [new_ICM_name]*len(new_row)
            new_data = np.array(counts[column], dtype=np.float32)

            item_original_ID_to_index_mapper = self._LOADED_GLOBAL_MAPPER_DICT['item_original_ID_to_index']
            ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index_mapper,
                                                            on_new_row="ignore")
            ICM_builder.add_data_lists(new_row, new_col, new_data)
            self.AVAILABLE_ICM.append(new_ICM_name)
            self.ICM_to_load_list.append(new_ICM_name)
            self._LOADED_ICM_DICT[new_ICM_name] = ICM_builder.get_SparseMatrix()
            self._LOADED_ICM_MAPPER_DICT[new_ICM_name] = ICM_builder.get_column_token_to_id_mapper()

        for UCM_name in self.UCM_names_to_ICM:
            UCM_object: sps.csr_matrix = self._LOADED_UCM_DICT[UCM_name]
            UCM_suffix_name = UCM_name.replace("UCM", "")

            new_ICM = self.get_URM_all().T.dot(UCM_object).tocoo()
            new_ICM_name = "ICM{}".format(UCM_suffix_name)
            vectorized_change_label = np.vectorize(lambda elem: "%s-%d" % (UCM_suffix_name, elem))

            new_row = np.array(new_ICM.row, dtype="str")
            new_col = vectorized_change_label(new_ICM.col)
            new_data = np.array(new_ICM.data, dtype=np.float32)

            item_original_ID_to_index_mapper = self._LOADED_GLOBAL_MAPPER_DICT['item_original_ID_to_index']
            ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index_mapper,
                                                            on_new_row="ignore")
            ICM_builder.add_data_lists(new_row, new_col, new_data)

            self.AVAILABLE_ICM.append(new_ICM_name)
            self.ICM_to_load_list.append(new_ICM_name)
            self._LOADED_ICM_DICT[new_ICM_name] = new_ICM
            self._LOADED_ICM_MAPPER_DICT[new_ICM_name] = ICM_builder.get_column_token_to_id_mapper()

        # Re-build ICM_all
        if "ICM_all" in self.get_loaded_ICM_names():
            self._LOADED_ICM_DICT["ICM_all"], self._LOADED_ICM_MAPPER_DICT["ICM_all"] = build_ICM_all(
                self._LOADED_ICM_DICT,
                self._LOADED_ICM_MAPPER_DICT)


class DataPreprocessingImputation(AbstractDataPreprocessing):

    DATASET_SUBFOLDER = "i/"

    def __init__(self, reader: DataReader, ICM_name_to_agg_mapper: dict, soft_copy=True):
        """
        Preprocessing operation that imputes numerical values for ICMs

        :param reader: data reader
        :param ICM_name_to_agg_mapper: a mapper of ICM name to a function that takes as input an array and returns a
                                       value
        :param soft_copy:
        """
        super().__init__(reader, soft_copy)
        self.ICM_name_to_agg_mapper = ICM_name_to_agg_mapper

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        return URM_all

    def _preprocess_ICMs(self):
        if ~np.all(np.in1d(list(self.ICM_name_to_agg_mapper.keys()), list(self._LOADED_ICM_DICT.keys()))):
            raise KeyError("Mapper contains wrong ICM names")

        for ICM_name, aggregator in self.ICM_name_to_agg_mapper.items():
            ICM_object: sps.csr_matrix = self._LOADED_ICM_DICT[ICM_name]
            x = np.array(ICM_object.data)
            missing_x_mask = np.ediff1d(ICM_object.tocsr().indptr) == 0
            missing_x = np.arange(ICM_object.shape[0])[missing_x_mask]
            new_row = np.concatenate([ICM_object.tocoo().row, missing_x])
            new_col = np.concatenate([ICM_object.tocoo().col, [ICM_object.indices[0]] * len(missing_x)])
            new_data = np.concatenate([ICM_object.data, [aggregator(x)] * len(missing_x)])

            vectorized_change_label = np.vectorize(lambda elem: "%s-%d" % (ICM_name, elem))
            new_col = vectorized_change_label(new_col)

            item_original_ID_to_index_mapper = self._LOADED_GLOBAL_MAPPER_DICT['item_original_ID_to_index']
            ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index_mapper,
                                                            on_new_row="ignore")
            ICM_builder.add_data_lists(np.array(new_row, dtype=str), new_col, np.array(new_data, dtype=np.float32))
            self._LOADED_ICM_DICT[ICM_name] = ICM_builder.get_SparseMatrix()
            self._LOADED_ICM_MAPPER_DICT[ICM_name] = ICM_builder.get_column_token_to_id_mapper()

        # Re-build ICM_all
        if "ICM_all" in self.get_loaded_ICM_names():
            self._LOADED_ICM_DICT["ICM_all"], self._LOADED_ICM_MAPPER_DICT["ICM_all"] = build_ICM_all(
                self._LOADED_ICM_DICT,
                self._LOADED_ICM_MAPPER_DICT)


class DataPreprocessingTransform(AbstractDataPreprocessing):

    DATASET_SUBFOLDER = "t/"

    def __init__(self, reader: DataReader, ICM_name_to_transform_mapper: dict, soft_copy=True):
        """
        Transform ICM data by a function given in the mapper
        :param reader: data reader
        :param ICM_name_to_transform_mapper: A mapper from ICM name to a function that takes as input an array and
                                             give as output another array with the same size
        :param soft_copy:
        """
        super().__init__(reader, soft_copy)
        self.ICM_name_to_transform_mapper = ICM_name_to_transform_mapper

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        return URM_all

    def _preprocess_ICMs(self):
        if ~np.all(np.in1d(list(self.ICM_name_to_transform_mapper.keys()), list(self._LOADED_ICM_DICT.keys()))):
            raise KeyError("Mapper contains wrong ICM names")

        for ICM_name, transformer in self.ICM_name_to_transform_mapper.items():
            ICM_object: sps.csr_matrix = self._LOADED_ICM_DICT[ICM_name]

            ICM_object.data = transformer(ICM_object.data)
            self._LOADED_ICM_DICT[ICM_name] = ICM_object

        # Re-build ICM_all
        if "ICM_all" in self.get_loaded_ICM_names():
            self._LOADED_ICM_DICT["ICM_all"], self._LOADED_ICM_MAPPER_DICT["ICM_all"] = build_ICM_all(
                self._LOADED_ICM_DICT,
                self._LOADED_ICM_MAPPER_DICT)


class DataPreprocessingDiscretization(AbstractDataPreprocessing):
    DATASET_SUBFOLDER = "d/"

    def __init__(self, reader: DataReader, ICM_name_to_bins_mapper: dict, soft_copy=True):
        """
        Digitize ICMs and UCMs whose name is inside the ICM_name_to_bins_mapper and UCM_name_to_bins_mapper
        and re-merge ICMs/UCMs in order to obtain ICM_all/UCM_all

        :param reader: data reader
        :param ICM_name_to_bins_mapper: mapper from ICM_name to number of bins
        :param soft_copy

        """
        super().__init__(reader, soft_copy)
        self.ICM_name_to_bins_mapper = ICM_name_to_bins_mapper

    def _preprocess_URM_all(self, URM_all: sps.csr_matrix):
        return URM_all

    def _preprocess_ICMs(self):
        if ~np.all(np.in1d(list(self.ICM_name_to_bins_mapper.keys()), list(self._LOADED_ICM_DICT.keys()))):
            raise KeyError("Mapper contains wrong ICM names")

        # Digitize unskewed data of ICMs
        for ICM_name, bins in self.ICM_name_to_bins_mapper.items():
            ICM_object: sps.csr_matrix = self._LOADED_ICM_DICT[ICM_name]
            if ICM_object.shape[1] != 1:
                raise KeyError("Given ICM name is not regarding a single feature, thus, it cannot be discretized")

            x = np.array(ICM_object.data)
            labelled_x = transform_numerical_to_label(x, bins)
            vectorized_change_label = np.vectorize(lambda elem: "%s-%d" % (ICM_name, elem))
            labelled_x = vectorized_change_label(labelled_x)

            item_original_ID_to_index_mapper = self._LOADED_GLOBAL_MAPPER_DICT['item_original_ID_to_index']
            ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index_mapper,
                                                            on_new_row="ignore")
            ICM_builder.add_data_lists(np.array(ICM_object.tocoo().row, dtype=str), labelled_x,
                                       np.ones(len(labelled_x), dtype=np.float32))

            self._LOADED_ICM_DICT[ICM_name] = ICM_builder.get_SparseMatrix()
            self._LOADED_ICM_MAPPER_DICT[ICM_name] = ICM_builder.get_column_token_to_id_mapper()

        # Re-build ICM_all
        if "ICM_all" in self.get_loaded_ICM_names():
            self._LOADED_ICM_DICT["ICM_all"], self._LOADED_ICM_MAPPER_DICT["ICM_all"] = build_ICM_all(
                self._LOADED_ICM_DICT,
                self._LOADED_ICM_MAPPER_DICT)
