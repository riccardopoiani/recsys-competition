import os

from course_lib.Base.DataIO import DataIO
from course_lib.Data_manager.DataReader import DataReader
from src.data_management.RecSys2019Reader_utils import load_ICM_sub_class, load_ICM_asset, load_ICM_price, \
    load_ICM_item_pop, load_URM, load_UCM_age, load_UCM_region, load_UCM_user_act, build_ICM_all


class RecSys2019Reader(DataReader):
    DATASET_SUBFOLDER = "data/"
    AVAILABLE_ICM = ["ICM_all", "ICM_price", "ICM_asset", "ICM_sub_class", "ICM_item_pop"]
    AVAILABLE_UCM = ["UCM_age", "UCM_region", "UCM_user_act", "UCM_all"]
    AVAILABLE_URM = ["URM_all"]

    _LOADED_UCM_DICT = None
    _LOADED_UCM_MAPPER_DICT = None

    IS_IMPLICIT = True

    def __init__(self, root_path="../data/", reload_from_original=False):
        super().__init__(reload_from_original_data=reload_from_original)
        self.root_path = root_path
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

    def get_loaded_UCM_dict(self):
        UCM_dict = {}

        for UCM_name in self.get_loaded_UCM_names():
            UCM_dict[UCM_name] = self.get_UCM_from_name(UCM_name)

        return UCM_dict

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

        print("RecSys2019Reader: loading ICM subclass, asset, price and item_pop")
        ICM_sub_class, tokenToFeatureMapper_ICM_sub_class, self.item_original_ID_to_index = load_ICM_sub_class(
            self.ICM_sub_class_path,
            separator=',')

        self._LOADED_ICM_DICT["ICM_sub_class"] = ICM_sub_class
        self._LOADED_ICM_MAPPER_DICT["ICM_sub_class"] = tokenToFeatureMapper_ICM_sub_class

        ICM_asset, tokenToFeatureMapper_ICM_asset, self.item_original_ID_to_index = load_ICM_asset(self.ICM_asset_path,
                                                                                                   separator=',',
                                                                                                   if_new_item="ignore",
                                                                                                   item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_asset"] = ICM_asset
        self._LOADED_ICM_MAPPER_DICT["ICM_asset"] = tokenToFeatureMapper_ICM_asset

        ICM_price, tokenToFeatureMapper_ICM_price, self.item_original_ID_to_index = load_ICM_price(self.ICM_price_path,
                                                                                                   separator=',',
                                                                                                   if_new_item="ignore",
                                                                                                   item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_price"] = ICM_price
        self._LOADED_ICM_MAPPER_DICT["ICM_price"] = tokenToFeatureMapper_ICM_price

        ICM_item_pop, tokenToFeatureMapper_ICM_item_pop, self.item_original_ID_to_index = load_ICM_item_pop(self.URM_path,
                                                                                                            separator=',',
                                                                                                            if_new_item="ignore",
                                                                                                            item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_ICM_DICT["ICM_item_pop"] = ICM_item_pop
        self._LOADED_ICM_MAPPER_DICT["ICM_item_pop"] = tokenToFeatureMapper_ICM_item_pop

        print("RecSys2019Reader: loading URM")

        URM_all, self.item_original_ID_to_index, self.user_original_ID_to_index = load_URM(self.URM_path, separator=",",
                                                                                           if_new_item="ignore",
                                                                                           item_original_ID_to_index=self.item_original_ID_to_index)
        self._LOADED_URM_DICT["URM_all"] = URM_all
        self._LOADED_GLOBAL_MAPPER_DICT["user_original_ID_to_index"] = self.user_original_ID_to_index
        self._LOADED_GLOBAL_MAPPER_DICT["item_original_ID_to_index"] = self.item_original_ID_to_index

        print("RecSys2019Reader: loading UCM age, region and user_act")
        print("RecSys2019Reader: WARNING --> There is no verification in the consistency of UCMs")

        UCM_age, tokenToFeatureMapper_UCM_age, self.user_original_ID_to_index = load_UCM_age(self.UCM_age_path,
                                                                                             separator=",",
                                                                                             if_new_user="ignore",
                                                                                             user_original_ID_to_index=self.user_original_ID_to_index)

        self._LOADED_UCM_DICT["UCM_age"] = UCM_age
        self._LOADED_UCM_MAPPER_DICT["UCM_age"] = tokenToFeatureMapper_UCM_age

        UCM_region, tokenToFeatureMapper_UCM_region, self.user_original_ID_to_index = load_UCM_region(
            self.UCM_region_path,
            separator=",",
            if_new_user="ignore",
            user_original_ID_to_index=self.user_original_ID_to_index)

        self._LOADED_UCM_DICT["UCM_region"] = UCM_region
        self._LOADED_UCM_MAPPER_DICT["UCM_region"] = tokenToFeatureMapper_UCM_region

        UCM_user_act, tokenToFeatureMapper_UCM_user_act, self.user_original_ID_to_index = load_UCM_user_act(
            self.URM_path,
            separator=",",
            if_new_user="ignore",
            user_original_ID_to_index=self.user_original_ID_to_index)

        self._LOADED_UCM_DICT["UCM_user_act"] = UCM_user_act
        self._LOADED_UCM_MAPPER_DICT["UCM_user_act"] = tokenToFeatureMapper_UCM_user_act

        print("RecSys2019Reader: loading ICM all")

        self._LOADED_ICM_DICT["ICM_all"], self._LOADED_ICM_MAPPER_DICT["ICM_all"] = build_ICM_all(self._LOADED_ICM_DICT,
                                                                                                  self._LOADED_ICM_MAPPER_DICT)

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

