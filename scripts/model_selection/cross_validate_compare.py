from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import *
from src.data_management.New_DataSplitter_leave_k_out import *
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.RecSys2019Reader_utils import get_ICM_numerical
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import best_models, new_best_models


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


def _get_all_models_ranked(URM_train, ICM_all, UCM_all, ICM_subclass_all):
    all_models = {}

    all_models['MIXED'] = best_models.WeightedAverageMixed.get_model(URM_train=URM_train,
                                                                     ICM_all=ICM_all,
                                                                     ICM_subclass_all=ICM_subclass_all,
                                                                     UCM_age_region=UCM_all)

    all_models['SLIM_BPR'] = best_models.SLIM_BPR.get_model(URM_train)
    all_models['P3ALPHA'] = best_models.P3Alpha.get_model(URM_train)
    all_models['RP3BETA'] = best_models.RP3Beta.get_model(URM_train)
    all_models['IALS'] = best_models.IALS.get_model(URM_train)
    all_models['USER_ITEM_ALL'] = best_models.UserItemKNNCBFCFDemographic.get_model(URM_train, ICM_all, UCM_all)

    return all_models

if __name__ == '__main__':
    seed_list = [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467, 6, 151, 86, 99, 4444]

    new_best_param = {'topK': 7, 'shrink': 1494, 'similarity': 'cosine', 'normalize': True,
                      'feature_weighting': 'TF-IDF', 'interactions_feature_weighting': 'TF-IDF'}
    model_score = 0

    for i in range(0, len(seed_list)):
        # Data loading
        root_data_path = "../data/"
        data_reader = RecSys2019Reader(root_data_path)
        data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                                   force_new_split=True, seed=seed_list[i])
        data_reader.load_data()
        URM_train, URM_test = data_reader.get_holdout_split()

        # Build ICMs
        ICM_all = get_ICM_train(data_reader)

        # Build UCMs: do not change the order of ICMs and UCMs
        UCM_all = get_UCM_train(data_reader, root_data_path)
        # Build UCMs

        # Setting evaluator
        cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
        cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
        warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) < 80
        warm_users = np.arange(URM_train.shape[0])[warm_users_mask]
        ignore_users = np.unique(np.concatenate((cold_users, warm_users)))

        cutoff_list = [10]
        evaluator_total = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=cold_users)

        # Building the models
        fusion = new_best_models.FusionMergeItem_CBF_CF.get_model(URM_train, ICM_all)

        curr_map = evaluator_total.evaluateRecommender(fusion)[0][10]['MAP']

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
