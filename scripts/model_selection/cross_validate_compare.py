from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.data_management.data_reader import get_ICM_train, get_UCM_train, get_ICM_train_new
from src.model import best_models, new_best_models
from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender
from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender


def _get_all_models_weighted_average(URM_train, ICM_all, UCM_all, ICM_subclass_all, ICM_numerical, ICM_categorical):
    all_models = {}

    all_models['ITEMCBFALLFOL'] = best_models.ItemCBF_all_EUC1_FOL3.get_model(URM_train=URM_train,
                                                                              ICM_train=ICM_all,
                                                                              load_model=False)
    all_models['ITEMCBFCFFOL'] = best_models.ItemCBF_CF_FOL_3_ECU_1.get_model(URM_train=URM_train,
                                                                              ICM_train=ICM_subclass_all,
                                                                              load_model=False)
    all_models['ITEMCFFOL'] = best_models.ItemCF_EUC_1_FOL_3.get_model(URM_train=URM_train)
    return all_models


def _get_all_models_ranked(URM_train, ICM_all, UCM_all):
    all_models = {}

    all_models['MIXED'] = new_best_models.MixedItem.get_model(URM_train, ICM_all)

    all_models['S_SLIM_BPR'] = new_best_models.SSLIM_BPR.get_model(sps.vstack([URM_train, ICM_all.T]))
    all_models['S_PURE_SVD'] = new_best_models.PureSVDSideInfo.get_model(URM_train, ICM_all)
    all_models['S_IALS'] = new_best_models.IALSSideInfo.get_model(URM_train, ICM_all)
    all_models['USER_CBF_CF'] = new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all)

    return all_models


if __name__ == '__main__':
    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 203]

    new_best_param = {'MIXED': 0.9905701189936175,
                      'S_SLIM_BPR': 0.03388611712880397, 'S_PURE_SVD': 0.016614177533037847,
                      'S_IALS': 0.02333406947854522, 'USER_CBF_CF': 0.07909998116549612}
    model_score = 0

    for i in range(0, len(seed_list)):
        # Data loading
        root_data_path = "../../data/"
        data_reader = RecSys2019Reader(root_data_path)
        data_reader = DataPreprocessingRemoveColdUsersItems(data_reader, threshold_users=20, threshold_items=-1)
        data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=1, use_validation_set=False,
                                                   force_new_split=True, seed=seed_list[i])
        data_reader.load_data()
        URM_train, URM_test = data_reader.get_holdout_split()

        # Build ICMs
        ICM_all, _ = get_ICM_train_new(data_reader)

        # Build UCMs: do not change the order of ICMs and UCMs
        UCM_all = get_UCM_train(data_reader)

        # Setting evaluator
        cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
        warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) < 80
        warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
        ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

        cutoff_list = [10]
        evaluator_total = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

        model = new_best_models.FusionMergeItem_CBF_CF.get_model(URM_train, ICM_all)

        curr_map = evaluator_total.evaluateRecommender(model)[0][10]['MAP']

        print("SEED: {} \n ".format(seed_list[i]))
        print("CURR MAP {} \n ".format(curr_map))

        model_score += curr_map

    model_score /= len(seed_list)

    # Store results on file
    destination_path = "../../report/cross_validation/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    destination_path = destination_path + "cross_valid_comparison" + now + ".txt"
    num_folds = len(seed_list)

    f = open(destination_path, "w")
    f.write("X-validation ItemCBFCF new best model FOH 80\n\n")

    f.write("X-val map {} \n\n".format(model_score))

    f.write("Number of folds: " + str(num_folds) + "\n")
    f.write("Seed list: " + str(seed_list) + "\n\n")

    f.close()
