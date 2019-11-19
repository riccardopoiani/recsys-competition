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
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sps
import os
from numpy.random import seed

import sys
sys.path.append("..")

from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.model_management.model_result_reader import best_model_reader
from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
from src.data_management.DataPreprocessing import DataPreprocessingRemoveColdUsersItems
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out

# +
SEED = 69420
seed(SEED)

# Data loading

dataset = RecSys2019Reader("../data/")
dataset = DataPreprocessingRemoveColdUsersItems(dataset, threshold_users=3)
dataset = New_DataSplitter_leave_k_out(dataset, k_out_value=3, use_validation_set=False, force_new_split=True)
dataset.load_data()

seed() # reset random seeds for other things
# -

URM_train, URM_test = dataset.get_holdout_split()
ICM_all = dataset.get_ICM_from_name('ICM_all')

best_model_list = best_model_reader("../report/hp_tuning/item_cbf/Nov19_11-23-21_k_out_value_3/")

best_model_list

cosine_best_model = ItemKNNCBFRecommender(ICM_all, URM_train)
cosine_best_model.fit(topK=9, shrink=968, similarity='cosine', normalize=True, feature_weighting='TF-IDF')

from src.model_management.evaluator import evaluate_recommender_by_user_demographic
from src.plots.plot_evaluation_helper import plot_metric_results_by_user_demographic

# ## User activity

user_activity = (URM_train > 0).sum(axis=1)
user_activity = np.array(user_activity).squeeze()
user_activity

results = evaluate_recommender_by_user_demographic(cosine_best_model, URM_train, URM_test, cutoff_list=[10], 
                                         user_demographic=user_activity, n_folds=20)

plot_metric_results_by_user_demographic(results, user_demographic=user_activity, user_demographic_name="User activity")

# ## Age

from src.data_management.data_getter import get_user_demographic

# +
reader = RecSys2019Reader("../data/")
reader.load_data()

URM_all = reader.get_URM_all()
UCM_age = reader.get_UCM_from_name('UCM_age')

age_demographic = get_user_demographic(UCM_age, URM_all, 3)
# -

results = evaluate_recommender_by_user_demographic(cosine_best_model, URM_train, URM_test, cutoff_list=[10], 
                                         user_demographic=age_demographic, n_folds=20)

plot_metric_results_by_user_demographic(results, user_demographic=age_demographic, user_demographic_name="Age")

# ## Region

# +
UCM_region = reader.get_UCM_from_name('UCM_region')

region_demographic = get_user_demographic(UCM_region, URM_all, 3)
# -

results = evaluate_recommender_by_user_demographic(cosine_best_model, URM_train, URM_test, cutoff_list=[10], 
                                         user_demographic=region_demographic, n_folds=20)

plot_metric_results_by_user_demographic(results, user_demographic=region_demographic, user_demographic_name="Region")


