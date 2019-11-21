from course_lib.Data_manager.DataReader_utils import merge_ICM
from src.plots.recommender_plots import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.New_DataSplitter_leave_k_out import *
from course_lib.KNN.ItemKNNCBFRecommender import *

if __name__ == '__main__':
    data_reader = RecSys2019Reader()
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_price = data_reader.get_ICM_from_name("ICM_price")
    ICM_asset = data_reader.get_ICM_from_name("ICM_asset")
    ICM_item_pop = data_reader.get_ICM_from_name("ICM_item_pop")
    ICM_numerical, _ = merge_ICM(ICM_price, ICM_asset, {}, {})
    ICM_numerical, _ = merge_ICM(ICM_numerical, ICM_item_pop, {}, {})

    path = "../report/hp_tuning/item_cbf_price_asset_itempop/Nov21_22-05-53_k_out_value_3/"
    basic_plots_from_tuning_results(path, ItemKNNCBFRecommender, URM_train, URM_test, save_on_file=True,
                                    output_path_folder="../report/graphics/item_cbf_price_asset_itempop/Nov21_22-05-53_k_out_value_3/",
                                    ICM=ICM_numerical)
