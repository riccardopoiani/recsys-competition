
import pandas as pd
import numpy as np
import os
from course_lib.Data_manager.DataReader import DataReader
from course_lib.Data_manager.DataReader_utils import merge_ICM

def _loadURM(filePath, separator=",", if_new_user = "add", if_new_item = "add",
                                     item_original_ID_to_index = None,
                                     user_original_ID_to_index = None):
    from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    URM_all_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_col_mapper = item_original_ID_to_index,
                                                    on_new_col = if_new_item,
                                                    preinitialized_row_mapper = user_original_ID_to_index,
                                                    on_new_row = if_new_user)

    df_original = pd.read_csv(filepath_or_buffer=filePath, sep=separator,
                              usecols=['playlist_id', 'track_id'],
                              dtype={'playlist_id': str, 'track_id': str})

    user_id_list = df_original['playlist_id'].values
    item_id_list = df_original['track_id'].values
    rating_list = np.ones(len(user_id_list), dtype=np.float64)

    URM_all_builder.add_data_lists(user_id_list, item_id_list, rating_list)
    return URM_all_builder.get_SparseMatrix(), \
           URM_all_builder.get_column_token_to_id_mapper(), \
           URM_all_builder.get_row_token_to_id_mapper()


def _loadICM_album(filePath, separator=",", if_new_item = "add", item_original_ID_to_index = None):
    from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(filePath, sep=separator,
                     usecols=['track_id', 'album_id'],
                     dtype={'track_id': str, 'album_id': str})

    df_original['album_id'] = "album-id-" + df_original['album_id']

    track_id_list = df_original['track_id'].values
    album_id_list = df_original['album_id'].values

    ICM_builder.add_data_lists(track_id_list, album_id_list, np.ones(len(track_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


def _loadICM_artist(filePath, separator=',', if_new_item = "add", item_original_ID_to_index = None):
    from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix_FilterIDs

    ICM_builder = IncrementalSparseMatrix_FilterIDs(preinitialized_row_mapper=item_original_ID_to_index,
                                                    on_new_row=if_new_item)

    df_original = pd.read_csv(filePath, sep=separator,
                     usecols=['track_id', 'artist_id'],
                     dtype={'track_id': str, 'artist_id': str})

    df_original['artist_id'] = "artist_id-" + df_original['artist_id']

    track_id_list = df_original['track_id'].values
    artist_id_list = df_original['artist_id'].values

    ICM_builder.add_data_lists(track_id_list, artist_id_list, np.ones(len(track_id_list), dtype=np.float64))

    return ICM_builder.get_SparseMatrix(), \
           ICM_builder.get_column_token_to_id_mapper(), \
           ICM_builder.get_row_token_to_id_mapper()


class RecSys2018Reader(DataReader):
    DATASET_SUBFOLDER = "data/"
    AVAILABLE_ICM = ["ICM_all", "ICM_album", "ICM_artist"]
    AVAILABLE_URM = ["URM_all"]
    IS_IMPLICIT = True

    def __init__(self, URM_path: os.path, ICM_path: os.path):
        super().__init__()
        self.URM_path = URM_path
        self.ICM_path = ICM_path

    def _get_dataset_name_root(self):
        return self.DATASET_SUBFOLDER

    def _load_from_original_file(self):
        # Load data from original file

        print("Recsys2018Reader: Loading original data")

        print("Recsys2018Reader: loading ICM album and artist")

        ICM_album, tokenToFeatureMapper_ICM_album, self.item_original_ID_to_index = _loadICM_album(self.ICM_path,
                                                                                                   separator=',',)
        self._LOADED_ICM_DICT["ICM_album"] = ICM_album
        self._LOADED_ICM_MAPPER_DICT["ICM_album"] = tokenToFeatureMapper_ICM_album

        ICM_artist, tokenToFeatureMapper_ICM_artist, self.item_original_ID_to_index = _loadICM_artist(self.ICM_path,
                                                                                                      separator=',',
                                                                                                      if_new_item="ignore",
                                                                                                      item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_artist"] = ICM_artist
        self._LOADED_ICM_MAPPER_DICT["ICM_artist"] = tokenToFeatureMapper_ICM_artist

        print("Recsys2018Reader: loading URM")

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = _loadURM(self.URM_path, separator=",",
                                                                                           if_new_item="ignore",
                                                                                           item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print("Recsys2018Reader: loading ICM all")

        ICM_all, tokenToFeatureMapper_ICM_all = merge_ICM(ICM_artist, ICM_album, tokenToFeatureMapper_ICM_artist,
                                                          tokenToFeatureMapper_ICM_album)

        self._LOADED_ICM_DICT["ICM_all"] = ICM_all
        self._LOADED_ICM_MAPPER_DICT["ICM_all"] = tokenToFeatureMapper_ICM_all

        print("Recsys2018Reader: loading complete")