# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# +
# Avoid users for which you have UCM info
mask_ucm = np.ediff1d(UCM_all.tocsr().indptr) > 0
ucm_warm = np.arange(UCM_all.shape[0])[mask_ucm]

# Avoid users for which you have URM info
warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
warm_users = np.arange(URM_train.shape[0])[warm_users_mask]

# Merge them
ignore_users = np.concatenate((warm_users, ucm_warm))
ignore_users = np.unique(ignore_users)

# Evaluate
evaluator_ucm_cold = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)
evaluator_ucm_cold.evaluateRecommender(usercbf)[0][10]['MAP']

# +
from datetime import datetime

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader, merge_ICM
from src.model.best_models import ItemCF
from src.tuning.run_parameter_search_cfw_linalg import run_parameter_search
from src.utils.general_utility_functions import get_split_seed
from src.data_management.RecSys2019Reader_utils import merge_UCM
from src.data_management.data_getter import get_warmer_UCM
import numpy as np

from src.model.best_models import UserCBF_CF

# + {"pycharm": {"is_executing": false}, "cell_type": "markdown"}
# # Cold-start current methods analysis
# So far, we have a winner for cold users looking at their MAP in general.
# However, we would like to understand if -when no UCM information is present- a top-popular recommender might score better. 

# +
# Data loading
data_reader = RecSys2019Reader("../../data/")
data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())

data_reader.load_data()
URM_train, URM_test = data_reader.get_holdout_split()
URM_all = data_reader.dataReader_object.get_URM_all()
UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

#UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
#UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})
URM_all, _ = merge_UCM(UCM_age_region, URM_all, {}, {})

# +
warm_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 0
warm_users = np.arange(URM_train.shape[0])[warm_users_mask]

# Setting evaluator
cutoff_list = [10]
evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=warm_users)
# -

usercbf = UserCBF_CF.get_model(URM_train, UCM_all)

evaluator.evaluateRecommender(usercbf)[0][10]['MAP']

# As we can see, generally speaking, we have a higher MAP results for all the cold users... but for some of them we have information in the UCM. How the recommender behaves for those that do not have such information? In principle, it should consider the user_profile_len of 0, as the only information that it has...

# ### How can we do it? Well using the readers, for all the ones that we have ratings, we have also information in the UCM... Therefore, we will take away UCM information for some cold users

# +
# Data loading
data_reader = RecSys2019Reader("../../data/")
data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())

data_reader.load_data()
URM_train, URM_test = data_reader.get_holdout_split()
URM_all = data_reader.dataReader_object.get_URM_all()
UCM_age = data_reader.dataReader_object.get_UCM_from_name("UCM_age")
UCM_region = data_reader.dataReader_object.get_UCM_from_name("UCM_region")
UCM_age_region, _ = merge_UCM(UCM_age, UCM_region, {}, {})

UCM_age_region = get_warmer_UCM(UCM_age_region, URM_all, threshold_users=3)
#UCM_all, _ = merge_UCM(UCM_age_region, URM_train, {}, {})

# +
n_runs = 5
map_cumulated = 0
for i in range(0, n_runs):
    # Select a random subsets of the cold users
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    idx_users_to_take_out = np.random.randint(low=0, high=cold_users.size-1, size=int(cold_users.size*0.3)+100)
    idx_users_to_take_out = np.unique(idx_users_to_take_out)

    users_to_take_out = cold_users[idx_users_to_take_out]
    users_to_take_out.size
    temp = UCM_age_region.copy()
    temp[users_to_take_out] = 0
    temp.eliminate_zeros()
    UCM_all, _ = merge_UCM(temp, URM_train, {}, {})

    total_users = np.arange(URM_train.shape[0])
    ignore_users_mask = np.in1d(total_users, users_to_take_out, invert=True)
    ignore_users = total_users[ignore_users_mask]
    
    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)
    usercbf = UserCBF_CF.get_model(URM_train, UCM_all)
    res = evaluator.evaluateRecommender(usercbf)[0][10]['MAP']
    print(res)
    map_cumulated += res

print(map_cumulated/n_runs)
# -

# MAP is zero for these users! Let's counter prove this

# +
n_runs = 5
map_cumulated = 0
for i in range(0, n_runs):
    # Select a random subsets of the cold users
    cold_users_mask = np.ediff1d(URM_train.tocsr().indptr) == 0
    cold_users = np.arange(URM_train.shape[0])[cold_users_mask]
    idx_users_to_take_out = np.random.randint(low=0, high=cold_users.size-1, size=int(cold_users.size*0.3)+100)
    idx_users_to_take_out = np.unique(idx_users_to_take_out)

    users_to_take_out = cold_users[idx_users_to_take_out]
    users_to_take_out.size
    
    temp = UCM_age_region.copy()
    #temp[users_to_take_out] = 0
    #temp.eliminate_zeros()
    UCM_all, _ = merge_UCM(temp, URM_train, {}, {})

    total_users = np.arange(URM_train.shape[0])
    ignore_users_mask = np.in1d(total_users, users_to_take_out, invert=True)
    ignore_users = total_users[ignore_users_mask]
    
    # Setting evaluator
    cutoff_list = [10]
    evaluator = EvaluatorHoldout(URM_test, cutoff_list=cutoff_list, ignore_users=ignore_users)
    usercbf = UserCBF_CF.get_model(URM_train, UCM_all)
    res = evaluator.evaluateRecommender(usercbf)[0][10]['MAP']
    print(res)
    map_cumulated += res

print(map_cumulated/n_runs)
# -


