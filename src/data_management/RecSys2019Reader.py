import pandas as pd
import numpy as np
import os
from course_lib.Data_manager.DataReader import DataReader
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs


def _loadURM(filePath, separator=",", if_new_user="add", if_new_item="add",
             item_original_ID_to_index=None,
             user_original_ID_to_index=None):
    URM_all_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper=item_original_ID_to_index,
                                                        on_new_col=if_new_item,
                                                        preinitialized_row_mapper=user_original_ID_to_index,
                                                        on_new_row=if_new_user)

    df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    user_id_list = df_original['row'].values
    item_id_list = df_original['col'].values
    rating_list = np.ones(len(user_id_list), dtype=np.float64)

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)
    return URM_all_builder.get_SparseMatrix(), \
           URM_all_builder.get_column_token_to_id_mapper(), \
           URM_all_builder.get_row_token_to_id_mapper()


def _loadICM_asset(filePath, separator=",", if_new_item="add", item_original_ID_to_index=None):
    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(filePath, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    df_original['col'] = "asset-" + df_original['col']

    item_id_list = df_original['row'].values
    asset_id_list = df_original['col'].values

    ICM_builder.add_data_lists(item_id_list, asset_id_list, np.ones(len(item_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


def _loadICM_sub_class(filePath, separator=",", if_new_item="add", item_original_ID_to_index=None):
    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(filePath, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    df_original['col'] = "sub_class-" + df_original['col']

    item_id_list = df_original['row'].values
    sub_class_id_list = df_original['col'].values

    ICM_builder.add_data_lists(item_id_list, sub_class_id_list, np.ones(len(item_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


def _loadICM_price(filePath, separator=',', if_new_item="add", item_original_ID_to_index=None):
    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(filePath, sep=separator,
                              usecols=['row', 'col'],
                              dtype={'row': str, 'col': str})

    df_original['col'] = "price-" + df_original['col']

    item_id_list = df_original['row'].values
    price_id_list = df_original['col'].values

    ICM_builder.add_data_lists(item_id_list, price_id_list, np.ones(len(item_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


class RecSys2019Reader(DataReader):
    DATASET_SUBFOLDER = "data/"
    AVAILABLE_ICM = ["ICM_all", "ICM_price", "ICM_asset", "ICM_sub_class"]
    #AVAILABLE_UCM = ["UCM_ALL", "UCM_age", "UCM_region"]
    AVAILABLE_URM = ["URM_all"]
    IS_IMPLICIT = True

    def __init__(self, URM_path: os.path="../data/data_train.csv", ICM_asset_path: os.path="../data/data_ICM_asset.csv",
                 ICM_price_path: os.path="../data/data_ICM_price.csv", ICM_sub_class_path: os.path="../data/data_ICM_sub_class.csv"):
        super().__init__()
        self.URM_path = URM_path
        self.ICM_asset_path = ICM_asset_path
        self.ICM_price_path = ICM_price_path
        self.ICM_sub_class_path = ICM_sub_class_path

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original file

        print("Recsys2019Reader: Loading original data")

        print("Recsys2019Reader: loading ICM album and artist")

        ICM_asset, tokenToFeatureMapper_ICM_asset, self.item_original_ID_to_index = _loadICM_asset(self.ICM_asset_path,
                                                                                                   separator=',', )
        self._LOADED_ICM_DICT["ICM_asset"] = ICM_asset
        self._LOADED_ICM_MAPPER_DICT["ICM_asset"] = tokenToFeatureMapper_ICM_asset

        ICM_price, tokenToFeatureMapper_ICM_price, self.item_original_ID_to_index = _loadICM_price(self.ICM_price_path,
                                                                                                      separator=',',
                                                                                                      if_new_item="ignore",
                                                                                                      item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_price"] = ICM_price
        self._LOADED_ICM_MAPPER_DICT["ICM_price"] = tokenToFeatureMapper_ICM_price

        ICM_sub_class, tokenToFeatureMapper_ICM_sub_class, self.item_original_ID_to_index = _loadICM_sub_class(self.ICM_sub_class_path,
                                                                                                   separator=',',
                                                                                                   if_new_item="ignore",
                                                                                                   item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_sub_class"] = ICM_sub_class
        self._LOADED_ICM_MAPPER_DICT["ICM_sub_class"] = tokenToFeatureMapper_ICM_sub_class

        print("Recsys2019Reader: loading URM")

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM(self.URM_path, separator=",",
                                                                                           if_new_item="ignore",
                                                                                           item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print("Recsys2019Reader: loading ICM all")

        ICM_price_asset, tokenToFeatureMapper_ICM_price_asset = merge_ICM(ICM_price, ICM_asset,
                                                                          tokenToFeatureMapper_ICM_price,
                                                                          tokenToFeatureMapper_ICM_asset)

        ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(ICM_price_asset, ICM_sub_class,
                                                          tokenToFeatureMapper_ICM_price_asset,
                                                          tokenToFeatureMapper_ICM_sub_class)

        self._LOADED_ICM_DICT["ICM_all"] = ICM_all
        self._LOADED_ICM_MAPPER_DICT["ICM_all"] = tokenToFeatureMapper_ICM_all

        print("Recsys2019Reader: loading complete")
