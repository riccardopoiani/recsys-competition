from src.data_management.RecSys2019Reader import RecSys2019Reader
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from course_lib.ParameterTuning.run_parameter_search import runParameterSearch_Content
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from datetime import datetime
from course_lib.Data_manager.DataReader_utils import merge_ICM
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from numpy.random import seed
import scipy.sparse as sps

SEED = 69420

if __name__ == '__main__':
    # Set seed in order to have same splitting of data
    seed(SEED)

    # Data loading
    data_reader = RecSys2019Reader("../../data/")
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True)
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_price = data_reader.get_ICM_from_name("ICM_price")
    ICM_asset = data_reader.get_ICM_from_name("ICM_asset")
    ICM_item_pop = data_reader.get_ICM_from_name("ICM_item_pop")
    ICM_numerical, _ = merge_ICM(ICM_price, ICM_asset, {}, {})
    ICM_numerical, _ = merge_ICM(ICM_numerical, ICM_item_pop, {}, {})

    # Reset seed for hyper-parameter tuning
    seed()

    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list)

    # HP tuning
    print("Start tuning...")
    version_path = "../../report/hp_tuning/item_cbf_price_asset_itempop/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "_k_out_value_3/"
    version_path = version_path + now

    runParameterSearch_Content(URM_train=URM_train, ICM_object=ICM_numerical, ICM_name="ICM_numerical",
                               recommender_class=ItemKNNCBFRecommender,
                               evaluator_validation=evaluator,
                               metric_to_optimize="MAP",
                               output_folder_path=version_path,
                               similarity_type_list=['euclidean'],
                               parallelizeKNN=True,
                               n_cases=35)
    print("...tuning ended")
