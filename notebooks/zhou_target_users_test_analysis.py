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

# # Target Users Analysis

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.data_management.RecSys2019Reader import RecSys2019Reader

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
df_target = pd.read_csv("../data/data_target_users_test.csv")

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
df_target.head()

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
target_users = df_target.user_id.values
target_users

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are %d users in the target users"%len(target_users))
# -

# ## Analyze target users w.r.t. URM

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
dataset = RecSys2019Reader()
dataset.load_data()

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
URM_all = dataset.get_URM_all()
URM_all

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
URM_user_mapper = dataset.get_user_original_ID_to_index_mapper()
original_users_URM = list(URM_user_mapper.keys())

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
mask = np.in1d(target_users, original_users_URM, assume_unique=True)
missing_users = target_users[~mask]
missing_users

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are %d missing users w.r.t. URM users"%len(missing_users))
# -

# Are there users in URM but not in target test users?

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
mask = np.in1d(original_users_URM, target_users, assume_unique=True)
URM_users_but_not_target = np.array(original_users_URM)[~mask]
URM_users_but_not_target

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("Yes, there are %d users in URM but not in target"%len(URM_users_but_not_target))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## How many users can we train with Early Stopping (i.e. dropping validation set)? 
# Let's suppose we do a leave-3-out splitting of the training. How many users are still warm, i.e. still have a 
# sample in the training set?

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
user_activity = (URM_all > 0).sum(axis=1)
user_activity = np.array(user_activity).squeeze()
len(user_activity)

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
users_trainable = user_activity[user_activity > 3]
print("Basically, there are only %d users in the training set"%(len(users_trainable)))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Overall, since there are many users with a small user profile, dropping the validation set
# may loses the prediction of many users, but most of them were very difficult to predict with
# only so few data. So, still using a non-personalized recommender, clusterize those users in a group
# and recommend the same items of that group might be a solution or recommend very similar items to the
# small user profile

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## Analyze missing users in target with users demographics (i.e. age and region)

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ### Region

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
df_region = pd.read_csv("../data/data_UCM_region.csv")
df_region.head()

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
users_UCM_region = df_region.row.values

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
mask = np.in1d(missing_users, users_UCM_region, assume_unique=True)
missing_users_in_UCM_region = missing_users[~mask]
missing_users_in_UCM_region[:10]

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are still missing %d users considering UCM region"%len(missing_users_in_UCM_region))
print("Which means there are %d missing users that can be handled by UCM region"%(len(missing_users)-len(missing_users_in_UCM_region)))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ### Age

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
df_age = pd.read_csv("../data/data_UCM_age.csv")
df_age.head()

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
users_UCM_age = df_age.row.values

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
mask = np.in1d(missing_users, users_UCM_age, assume_unique=True)
missing_users_in_UCM_age = missing_users[~mask]
missing_users_in_UCM_age

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are still missing %d users considering UCM age"%len(missing_users_in_UCM_age))
print("Which means there are %d missing users that can be handled by UCM age"%(len(missing_users)-len(missing_users_in_UCM_age)))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ### Both Region and Age

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
users_UCM_age_region = np.intersect1d(users_UCM_age, users_UCM_region)
users_UCM_age_region

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
mask = np.in1d(missing_users, users_UCM_age_region, assume_unique=True)
missing_users_in_UCM_age_region = missing_users[~mask]
missing_users_in_UCM_age_region[:10]

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are still missing %d users considering UCM age"%len(missing_users_in_UCM_age_region))
print("Which means there are %d missing users that can be handled by UCM age"%(len(missing_users)-len(missing_users_in_UCM_age_region)))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## Conclusion

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Overall, using demographics user information, it is possible to cover at most 3027 users with both age and region
# while leaving out 629 users that can be handled by non-personalized recommender or by using only information about
# the age (even a hybrid if possible). Then, leaving out 77 users that can be handled by non-personalized recommender.
#
# Remember that these methods cannot be tested with validation set, because there is none. So, if you want it is only
# possible to test it via half-test set on Kaggle. This is just a gamble and you might overfit it when using information
# about demographics.
#
# Moreover, it is possible to use the demographics for the small profile users: those not trainable

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
users_not_trainable_mask = (user_activity > 0) & (user_activity <= 3)
users_not_trainable = np.array(original_users_URM)[users_not_trainable_mask]
users_not_trainable

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are %d users not trainable and not in missing users"%len(users_not_trainable))

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
mask = np.in1d(users_not_trainable, users_UCM_age_region, assume_unique=True)
users_not_trainable_in_UCM_age_region = users_not_trainable[~mask]
users_not_trainable_in_UCM_age_region

# + {"pycharm": {"name": "#%%\n", "is_executing": false}}
print("There are still %d users not trainable and not in UCM age and region"%len(users_not_trainable_in_UCM_age_region))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# So, even those not trainable users, i.e. that has $0 < user.activity <= 3$, are not covered by 
# the user demographics. 
#
# In conclusion, there are different mixture of ways to handle it:
#  - Using ICM and try to find similar items for small user profiles
#  - Using UCM (i.e. user demographics) to find a group of similar users (i.e. by clustering, ...) and provide
#  similar recommendations for users with empty user profiles
#  - Using non-personalized recommenders, i.e. Top-Popular (most easy way) for no user demographics
#
# Non-personalized recommender can always be considered in some way, i.e. hybrid with UCM or ICM techniques
#
