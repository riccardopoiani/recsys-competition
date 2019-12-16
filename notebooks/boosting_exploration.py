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

# + {"pycharm": {"is_executing": false}}
from datetime import datetime

import numpy as np
import pandas as pd

from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model.Ensemble.Boosting.boosting_preprocessing import add_label
from src.tuning.run_xgboost_tuning import run_xgb_tuning
from src.utils.general_utility_functions import get_split_seed
import matplotlib.pyplot as plt
from src.model import new_best_models
import xgboost as xgb
from xgboost import plot_importance

def _preprocess_dataframe(df: pd.DataFrame):
    df = df.copy()
    df = df.drop(columns=["index"], inplace=False)
    df = df.sort_values(by="user_id", ascending=True)
    df = df.reset_index()
    df = df.drop(columns=["index"], inplace=False)
    return df


# -

# # Data reading

# +
# Data loading
root_data_path = "../data/"
data_reader = RecSys2019Reader(root_data_path)
data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                           force_new_split=True, seed=get_split_seed())
data_reader.load_data()
URM_train, URM_test = data_reader.get_holdout_split()

# Build ICMs
ICM_all = get_ICM_train(data_reader)

# Build UCMs: do not change the order of ICMs and UCMs
UCM_all = get_UCM_train(data_reader)

# Reading the dataframe
dataframe_path = "../boosting_dataframe/"
train_df = pd.read_csv(dataframe_path + "train_df_20.csv")
valid_df = pd.read_csv(dataframe_path + "valid_df_20.csv")

train_df = _preprocess_dataframe(train_df)
valid_df = _preprocess_dataframe(valid_df)

print("Retrieving training labels...", end="")
y_train, non_zero_count, total = add_label(data_frame=train_df, URM_train=URM_train)
print("Done")

train_df['label'] = y_train
# -

print("Retrieving training labels...", end="")
y_train_valid, non_zero_count_vaid, total_valid = add_label(data_frame=valid_df, URM_train=URM_test)
print("Done")
valid_df['label'] = y_train_valid


# ### Pure scores exploration

def plot_score_distribution(column_name):
    # Let's check train and validation performances, minmaxstandardized
    train_mxd = train_df[column_name].values
    valid_mxd = valid_df[column_name].values

    # Sort them
    train_mxd = np.sort(train_mxd)
    valid_mxd = np.sort(valid_mxd)

    # Plot the distributions 
    plt.title('Distributions of {} scores between the two datasets'.format(column_name))
    plt.xlabel('(user, item) index')
    plt.ylabel('Score')
    plt.plot(train_mxd, label="Training")
    plt.plot(valid_mxd, label="Validation")
    plt.legend()
    plt.show()


plot_score_distribution("MixedItem")

plot_score_distribution("ItemCBF_CF")

plot_score_distribution("RP3BetaSideInfo")

plot_score_distribution("UserCF")

plot_score_distribution("ItemCBF_all_FW")


# As we can see, the "Validation" curve is always a bit above. This is due to how data has been generated: indeed, for training we keep the scores of the one with the highest score, removing the unseen with low score. Therefore, for many users, the one with few ratings, we remove many unseen items that are likely to be recommended.

def plot_score_distribution_unsorted(column_name_list, train=True):
    # Let's check train and validation performances, minmaxstandardized
    if train:
        df = train_df
    else:
        df = valid_df

    # Plot the distributions 
    plt.title("Distribution of scores unsorted")
    plt.xlabel('(user, item) index')
    plt.ylabel('Score')
    for name in column_name_list:
        plt.plot(df[name].values, label=name)
    plt.legend()
    plt.show()


name_list = ["MixedItem", "UserCF"]
plot_score_distribution_unsorted(name_list)

plot_score_distribution_unsorted(name_list, train=False)

# ### Scores mixed label exploration

# +
# Power users exploration
power_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 40
power_users = np.arange(URM_train.shape[0])[power_users_mask]

rare_users = np.ediff1d(URM_train.tocsr().indptr) < 7
rare_users = np.arange(URM_train.shape[0])[rare_users]

total_users = np.arange(URM_train.shape[0])
middle_users =  np.in1d(total_users, np.unique(np.concatenate((power_users, rare_users))), invert=True)
middle_users = total_users[middle_users]
                        
power_train_df = train_df[train_df['user_id'].isin(power_users)]
power_valid_df = valid_df[valid_df['user_id'].isin(power_users)]

rare_train_df = train_df[train_df['user_id'].isin(rare_users)]
rare_valid_df = valid_df[valid_df['user_id'].isin(rare_users)]
                        
middle_train_df = train_df[train_df['user_id'].isin(middle_users)]
middle_valid_df = valid_df[valid_df['user_id'].isin(middle_users)]
# -

power_train_df['label'].mean()

train_df['label'].mean()

rare_train_df['label'].mean()

valid_df['label'].mean()

power_valid_df['label'].mean()

rare_valid_df['label'].mean()

middle_valid_df['label'].mean()

middle_train_df['label'].mean()

# As we can see, there is no such a big difference between middle users and power users, in terms of mean of training labels that are present, in the validation set. Let's see if this is true even for the users with much ratings.

# +
extreme_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 100
extreme_users = np.arange(URM_train.shape[0])[extreme_users_mask]

extreme_train_df = train_df[train_df['user_id'].isin(extreme_users)]
extreme_valid_df = valid_df[valid_df['user_id'].isin(extreme_users)]
# -

extreme_train_df['label'].mean()

extreme_valid_df['label'].mean()

# Here we can notice that here we have something like an extreme situation: many 1 in train, and many 0s in test: the classifier may learn to predict only 1 when there are "large" user activities and 80% of predictions will be correct: moreover, remember also the unbalance you are giving to the loss function: miss-classifying 1s costs a lot! 
# Let's try to split things in group, and try to plot some graphics. Let's make some trial.

# +
boosting = new_best_models.Boosting.get_model(URM_train=URM_train, train_df=train_df, y_train=y_train,
                                                  valid_df=valid_df,
                                                  model_path="../report/hp_tuning/boosting/Dec15_12-21"
                                                             "-30_k_out_value_3_eval/best_model3")

boosting.RECOMMENDER_NAME = "BOOSTING"
    
# -

extreme_users[0:10]

user_38_valid_df = valid_df[valid_df['user_id']==38]
user_38_valid_df = user_38_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
boosting.bst.predict(xgb.DMatrix(user_38_valid_df))

# Indeed, that is exactly what is happening: the re-ranking might not working well for these users. Let's see if this is true for all the extreme users.

mean_extreme = []
for user in extreme_users:
    curr_user_valid_df = valid_df[valid_df['user_id']==user].copy()
    curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
    res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
    mean_extreme.append(res.mean())
    print("User {}: media {} - std {} - activity {}".format(user, res.mean(), res.std(), URM_train[user].indices.size))

plt.title("Distribution of mean predictions for extreme users")
plt.xlabel('power user index')
plt.ylabel('Mean prediction')
plt.plot(mean_extreme)
plt.show()

plt.title("Sorted Distribution of mean predictions for extreme users")
plt.xlabel('power user index')
plt.ylabel('Mean prediction')
plt.plot(np.sort(np.array(mean_extreme)))
plt.show()

# As we can see, 0s and 1s are not learnt correctly, since many of the average label for them is 0.01, while here we are getting 0.9: it is not learning the right things. How can we solve this issue, without introducing data leakage? 
# We have to re-think the way the dataset is generated. Let's check if also for middle users we have this behavior.

mean_middle = []
for user in middle_users:
    curr_user_valid_df = valid_df[valid_df['user_id']==user].copy()
    curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
    res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
    mean_middle.append(res.mean())

plt.title("Distribution of mean predictions for middle users")
plt.xlabel('Middle user index')
plt.ylabel('Mean predictions')
plt.plot(mean_middle)
plt.show()

plt.title("Sorted Distribution of mean predictions for middle users")
plt.xlabel('Middle user index')
plt.ylabel('Mean predictions')
plt.plot(np.sort(np.array(mean_middle)))
plt.show()

# +
power_users_not_extreme = np.in1d(power_users, extreme_users, invert=True)
power_users_not_extreme = power_users[power_users_not_extreme]

mean_power_not_extreme = []
for user in power_users_not_extreme:
    curr_user_valid_df = valid_df[valid_df['user_id']==user].copy()
    curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
    res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
    mean_power_not_extreme.append(res.mean())
    
plt.title("Distribution of mean predictions for power users not extreme")
plt.xlabel('User index')
plt.ylabel('Mean predictions')
plt.plot(mean_power_not_extreme)
plt.show()

plt.title("Distribution of mean predictions for power users not extreme")
plt.xlabel('User index')
plt.ylabel('Mean predictions')
plt.plot(np.sort(np.array(mean_power_not_extreme)))
plt.show()

# +
# Let's try to do this last step, sorting user due to their user activity profile
act_pune = URM_train[power_users_not_extreme].sum(axis=1)
act_pune = np.argsort(np.array(act_pune).squeeze())
power_users_not_extreme = power_users_not_extreme[act_pune]

mean_power_not_extreme_sorted = []
for user in power_users_not_extreme:
    curr_user_valid_df = valid_df[valid_df['user_id']==user].copy()
    curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
    res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
    mean_power_not_extreme_sorted.append(res.mean())
    
plt.title("Distribution of mean predictions for power users not extreme")
plt.xlabel('User index')
plt.ylabel('Mean predictions')
plt.plot(mean_power_not_extreme_sorted, label="Sorted By user act")
plt.plot(np.sort(np.array(mean_power_not_extreme)), label="sort by mean")
plt.legend()
plt.show()

# -

# Even if the lowest scores are descending, it still seems like the classifier is learning to return 1 too often... and this is true also for the middle users

plot_importance(boosting.bst)

# In order to trouble shoot this, let's better inspect the behavior for the middle users. Which users has such an high mean precitions? Let's say above 0.8

middle_high_mean = []
t = 0.8
for user in middle_users:
    curr_user_valid_df = valid_df[valid_df['user_id']==user].copy()
    curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
    res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
    if res.mean() > 0.8:
        middle_high_mean.append(user)

middle_users.size

middle_high_mean = np.array(middle_high_mean)
middle_high_mean.size

# For many of them we have such high behaviours...

middle_high_df_valid = valid_df[valid_df['user_id'].isin(middle_high_mean)]
middle_high_df_train = train_df[train_df['user_id'].isin(middle_high_mean)]

middle_high_df_valid['label'].mean()

middle_high_df_train['label'].mean()

# In the training set, we are almost in line with the mean of the training set... 

train_df['label'].mean()

middle_high_df_valid['ItemCBF_CF'].mean() # Score seems to be high 

middle_valid_df['ItemCBF_CF'].mean()

valid_df['ItemCBF_CF'].mean()

middle_high_df_valid['UserCF'].mean()

valid_df['UserCF'].mean()

middle_high_df_valid['UserCF'].mean()

middle_high_df.describe()

train_df.describe()

valid_df.describe()

# Scores in that part seems to be larger... maybe this could be a motivation, but it is still strange.
# The classifier is not behaving correctly.


