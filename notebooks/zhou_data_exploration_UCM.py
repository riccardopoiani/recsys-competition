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

# # EDA about UCM

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# ## Age

df_UCM_age = pd.read_csv("../data/data_UCM_age.csv")

df_UCM_age.head()

df_UCM_age.describe()

df_UCM_age['col'].value_counts().sort_index().plot.bar()

# ### Advanced plots with URM

df_URM = pd.read_csv("../data/data_train.csv")

df_URM.head()

# Let's see if the users in URM contains all the users in UCM_age or not

URM_users = df_URM.row.unique()
print("There are %d users in URM"%len(URM_users), end="")
UCM_age_users = df_UCM_age.row.unique()
print(", while there are %d users in UCM_age."%len(UCM_age_users))
mask = np.in1d(URM_users, UCM_age_users, assume_unique=True)
print("But there are %d users in URM and not in UCM_age"%len(URM_users[~mask]))

# #### Most active user
#
# Let's look into the most active users: 10% most active

user_activity = df_URM['row'].value_counts()
user_activity.describe(percentiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])

most_active_users = user_activity[user_activity >= user_activity.quantile(q=0.9)].index
most_active_users

df_UCM_age[df_UCM_age.row.isin(most_active_users)].col.value_counts().sort_index().plot.bar()

# It seems that by looking at the most active users, the distribution of the age is still the same. There is no correlation between most active users and age. Let's now see the least active users

# #### Least active users
# Let's look into the 40% least active users

least_active_users = user_activity[user_activity <= user_activity.quantile(q=0.4)].index
least_active_users

df_UCM_age[df_UCM_age.row.isin(least_active_users)].col.value_counts().sort_index().plot.bar()

# Overall, there is no big difference in the distribution of most active users and least active users between the overall distribution

# ## Region

df_UCM_region = pd.read_csv("../data/data_UCM_region.csv")

df_UCM_region.head()

df_UCM_region.describe()

df_UCM_region['col'].value_counts().sort_index().plot.bar()

# ### Advanced plots with URM

# #### Most active users

df_UCM_region[df_UCM_region.row.isin(most_active_users)].col.value_counts().sort_index().plot.bar()

# #### Least active users

df_UCM_region[df_UCM_region.row.isin(least_active_users)].col.value_counts().sort_index().plot.bar()

# Overall, it seems there is less users in region 4 when it is active, but more users in region 4 when it is inactive

# ## Age-Region

df_UCM_age_region = pd.merge(df_UCM_age, df_UCM_region, on='row')

# +
regions = df_UCM_age_region['col_y'].sort_values().unique()

fig, ax = plt.subplots(3, 2, figsize=(20, 25))

for i in range(len(regions)):
    df_UCM_age_region[df_UCM_age_region.col_y == regions[i]]['col_x'].value_counts().sort_index().plot.bar(ax=ax[i//2][i%2])
    ax[i//2][i%2].set_xlabel('Age')
    ax[i//2][i%2].set_ylabel('Frequency')
    ax[i//2][i%2].set_title('Region=%d'%regions[i])
# -

# By changing the region, the distribution of the age is still very similar to the original ones except some small changes.
