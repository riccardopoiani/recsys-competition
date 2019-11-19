import pandas as pd
import numpy as np
import os

from course_lib.Base.DataIO import DataIO
from course_lib.Data_manager.DataReader import DataReader
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs


def _loadURM(file_path, separator=",", if_new_user="add", if_new_item="add", item_original_ID_to_index=None,
             user_original_ID_to_index=None):
    URM_all_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_original_ID_to_index,
                                                        on_new_col=if_new_item,
                                                        preinitialized_row_mapper=user_original_ID_to_index,
                                                        on_new_row=if_new_user)

    df_original = pd.read_csv(filepath_or_buffer=file_path, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    user_id_list = df_original['row'].values
    item_id_list = df_original['col'].values
    rating_list = np.ones(len(user_id_list), dtype=np.float64)

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)
    return URM_all_builder.get_SparseMatrix(), \
           URM_all_builder.get_column_token_to_id_mapper(), \
           URM_all_builder.get_row_token_to_id_mapper()


def _loadICM_asset(file_path, separator=",", if_new_item="add", item_original_ID_to_index=None):
    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(file_path, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    df_original['col'] = "asset-" + df_original['col']

    item_id_list = df_original['row'].values
    asset_id_list = df_original['col'].values

    ICM_builder.add_data_lists(item_id_list, asset_id_list, np.ones(len(item_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


def _loadICM_sub_class(file_path, separator=",", if_new_item="add", item_original_ID_to_index=None):
    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(file_path, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    df_original['col'] = "sub_class-" + df_original['col']

    item_id_list = df_original['row'].values
    sub_class_id_list = df_original['col'].values

    ICM_builder.add_data_lists(item_id_list, sub_class_id_list, np.ones(len(item_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


def _loadICM_price(file_path, separator=',', if_new_item="add", item_original_ID_to_index=None):
    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(file_path, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    df_original['col'] = "price-" + df_original['col']

    item_id_list = df_original['row'].values
    price_id_list = df_original['col'].values

    ICM_builder.add_data_lists(item_id_list, price_id_list, np.ones(len(item_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


def _loadUCM_age(file_path, separator=",", if_new_user="add", user_original_ID_to_index=None):
    UCM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=user_original_ID_to_index,
                                                    on_new_row=if_new_user)

    df_original = pd.read_csv(file_path, sep=separator, usecols=['row', 'col'], dtype={'row': str, 'col': str})

    user_id_list = df_original['row'].values
    age_id_list = df_original['col'].values

    UCM_builder.add_data_lists(user_id_list, age_id_list, np.ones(len(user_id_list), dtype=np.float64))

    return UCM_builder.get_SparseMatrix(), UCM_builder.get_column_token_to_id_mapper(), \
           UCM_builder.get_row_token_to_id_mapper()


def _loadUCM_region(file_path, separator=",", if_new_user="add", user_original_ID_to_index=None):
    UCM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=user_original_ID_to_index,
                                                    on_new_row=if_new_user)

    df_original = pd.read_csv(file_path, sep=separator, usecols=['row', 'col'], dtype={'row': str, 'col': str})

    user_id_list = df_original['row'].values
    region_id_list = df_original['col'].values

    UCM_builder.add_data_lists(user_id_list, region_id_list, np.ones(len(user_id_list), dtype=np.float64))

    return UCM_builder.get_SparseMatrix(), UCM_builder.get_column_token_to_id_mapper(), \
           UCM_builder.get_row_token_to_id_mapper()


class RecSys2019Reader(DataReader):
    DATASET_SUBFOLDER = "data/"
    AVAILABLE_ICM = ["ICM_all", "ICM_price", "ICM_asset", "ICM_sub_class"]
    AVAILABLE_UCM = ["UCM_age", "UCM_region"]
    AVAILABLE_URM = ["URM_all"]

    _LOADED_UCM_DICT = None
    _LOADED_UCM_MAPPER_DICT = None

    IS_IMPLICIT = True

    def __init__(self, root_path="../data/", reload_from_original=False):
        super().__init__(reload_from_original_data=reload_from_original)
        self.URM_path = os.path.join(root_path, "data_train.csv")
        self.ICM_asset_path = os.path.join(root_path, "data_ICM_asset.csv")
        self.ICM_price_path = os.path.join(root_path, "data_ICM_price.csv")
        self.ICM_sub_class_path = os.path.join(root_path, "data_ICM_sub_class.csv")
        self.UCM_age_path = os.path.join(root_path, "data_UCM_age.csv")
        self.UCM_region_path = os.path.join(root_path, "data_UCM_region.csv")

        self._LOADED_UCM_DICT = {}
        self._LOADED_UCM_MAPPER_DICT = {}

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def get_UCM_from_name(self, UCM_name):
        self._assert_is_initialized()
        return self._LOADED_UCM_DICT[UCM_name].copy()

    def get_UCM_feature_to_index_mapper_from_name(self, UCM_name):
        self._assert_is_initialized()
        return self._LOADED_UCM_MAPPER_DICT[UCM_name].copy()

    def get_all_available_UCM_names(self):
        return self.AVAILABLE_UCM.copy()

    def get_loaded_UCM_names(self):
        return self.AVAILABLE_UCM.copy()

    def _load_from_original_file(self):
        # Load data from original file

        print("RecSys2019Reader: Loading original data")

        print("RecSys2019Reader: loading ICM subclass, asset and price")

        ICM_sub_class, tokenToFeatureMapper_ICM_sub_class, self.item_original_ID_to_index = _loadICM_sub_class(
            self.ICM_sub_class_path,
            separator=',', )

        self._LOADED_ICM_DICT["ICM_sub_class"] = ICM_sub_class
        self._LOADED_ICM_MAPPER_DICT["ICM_sub_class"] = tokenToFeatureMapper_ICM_sub_class

        ICM_asset, tokenToFeatureMapper_ICM_asset, self.item_original_ID_to_index = _loadICM_asset(self.ICM_asset_path,
                                                                                                   separator=',',
                                                                                                   if_new_item="ignore",
                                                                                                   item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_asset"] = ICM_asset
        self._LOADED_ICM_MAPPER_DICT["ICM_asset"] = tokenToFeatureMapper_ICM_asset

        ICM_price, tokenToFeatureMapper_ICM_price, self.item_original_ID_to_index = _loadICM_price(self.ICM_price_path,
                                                                                                   separator=',',
                                                                                                   if_new_item="ignore",
                                                                                                   item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_price"] = ICM_price
        self._LOADED_ICM_MAPPER_DICT["ICM_price"] = tokenToFeatureMapper_ICM_price

        print("RecSys2019Reader: loading URM")

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM(self.URM_path, separator=",",
                                                                                           if_new_item="ignore",
                                                                                           item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print("RecSys2019Reader: loading UCM age, region and all")
        print("RecSys2019Reader: WARNING --> There is no verification in the consistency of UCMs")
        print("RecSys2019Reader: WARNING --> The number of data in UCM is much lower than the original")
        print("RecSys2019Reader: WARNING --> UCM will be ignored by any preprocessing or data splitter class")

        UCM_age, tokenToFeatureMapper_UCM_age, self.user_original_ID_to_index = _loadUCM_age(self.UCM_age_path,
                                                                                             separator=",",
                                                                                             if_new_user="ignore",
                                                                                             user_original_ID_to_index=self.user_original_ID_to_index)

        self._LOADED_UCM_DICT["UCM_age"] = UCM_age
        self._LOADED_UCM_MAPPER_DICT["UCM_age"] = tokenToFeatureMapper_UCM_age

        UCM_region, tokenToFeatureMapper_UCM_region, self.user_original_ID_to_index = _loadUCM_region(
            self.UCM_region_path,
            separator=",",
            if_new_user="ignore",
            user_original_ID_to_index=self.user_original_ID_to_index)

        self._LOADED_UCM_DICT["UCM_region"] = UCM_region
        self._LOADED_UCM_MAPPER_DICT["UCM_region"] = tokenToFeatureMapper_UCM_region

        print("RecSys2019Reader: loading ICM all")

        ICM_price_asset, tokenToFeatureMapper_ICM_price_asset = merge_ICM(ICM_price, ICM_asset,
                                                                          tokenToFeatureMapper_ICM_price,
                                                                          tokenToFeatureMapper_ICM_asset)

        ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(ICM_price_asset, ICM_sub_class,
                                                          tokenToFeatureMapper_ICM_price_asset,
                                                          tokenToFeatureMapper_ICM_sub_class)

        self._LOADED_ICM_DICT["ICM_all"] = ICM_all
        self._LOADED_ICM_MAPPER_DICT["ICM_all"] = tokenToFeatureMapper_ICM_all

        print("RecSys2019Reader: loading complete")

    def _save_dataset(self, save_folder_path):
        """
        Saves all URM, ICM and UCM
        :param save_folder_path:
        :return:
        """
        dataIO = DataIO(folder_path=save_folder_path)

        dataIO.save_data(data_dict_to_save=self._LOADED_GLOBAL_MAPPER_DICT,
                         file_name="dataset_global_mappers")

        dataIO.save_data(data_dict_to_save=self._LOADED_URM_DICT,
                         file_name="dataset_URM")

        if len(self.get_loaded_ICM_names()) > 0:
            dataIO.save_data(data_dict_to_save=self._LOADED_ICM_DICT,
                             file_name="dataset_ICM")

            dataIO.save_data(data_dict_to_save=self._LOADED_ICM_MAPPER_DICT,
                             file_name="dataset_ICM_mappers")

        if len(self.get_loaded_UCM_names()) > 0:
            dataIO.save_data(data_dict_to_save=self._LOADED_UCM_DICT,
                             file_name="dataset_UCM")
            dataIO.save_data(data_dict_to_save=self._LOADED_UCM_MAPPER_DICT,
                             file_name="dataset_UCM_mappers")

    def _load_from_saved_sparse_matrix(self, save_folder_path):
        """
        Loads all URM, ICM and UCM
        :return:
        """
        dataIO = DataIO(folder_path=save_folder_path)
        self._LOADED_GLOBAL_MAPPER_DICT = dataIO.load_data(file_name="dataset_global_mappers")

        self._LOADED_URM_DICT = dataIO.load_data(file_name="dataset_URM")

        if len(self.get_loaded_ICM_names()) > 0:
            self._LOADED_ICM_DICT = dataIO.load_data(file_name="dataset_ICM")

            self._LOADED_ICM_MAPPER_DICT = dataIO.load_data(file_name="dataset_ICM_mappers")

        if len(self.get_loaded_UCM_names()) > 0:
            self._LOADED_UCM_DICT = dataIO.load_data(file_name="dataset_UCM")
            self._LOADED_UCM_MAPPER_DICT = dataIO.load_data(file_name="dataset_UCM_mappers")
            print("RecSys2019Reader: WARNING --> There is no verification in the consistency of UCMs")
            print("RecSys2019Reader: WARNING --> The number of data in UCM is much lower than the original")
            print("RecSys2019Reader: WARNING --> UCM will be ignored by any preprocessing or data splitter class")

