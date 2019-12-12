# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
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
from src.data_management.RecSys2019Reader import *
import pandas as pd
import numpy as np
import scipy.sparse as sps
import matplotlib.pyplot as plt
from src.data_management.data_getter import *
from src.plots.plot_evaluation_helper import * 
# -

# # Data reading

reader = RecSys2019Reader("../data/data_train.csv")
reader.load_data()

urm_all = reader.get_URM_all()

urm_all

URM_df = pd.read_csv('../data/data_train.csv')
URM_df.info()

# Some users seem to miss.
# Indeed, from the competition page we know that:
# The dataset includes around 426k interactions with 30911 users and 18495 items

# # URM plots

URM_df.head()

URM_df['row'].max()

URM_df['col'].max()

# Therefore, row are the users, while col are the items

user_list = URM_df['row']
item_list = URM_df['col']

user_list_unique = list(set(user_list))
item_list_unique = list(set(item_list))

# +
max_id_user = np.max(user_list_unique)
max_id_item = np.max(item_list_unique)

print("Max user ID: {}".format(max_id_user))
print("Max item ID: {}".format(max_id_item))
# -

print("Sparsity: {0:.4f}%".format(len(URM_df)/(len(user_list_unique)*len(item_list_unique))))

URM_all = sps.coo_matrix((np.ones(len(URM_df)), (user_list, item_list)))
URM_all = URM_all.tocsr()
URM_all

# # Item popularity

pop_items = (URM_all > 0).sum(axis=0)
pop_items = np.array(pop_items).squeeze()
pop_items = np.sort(pop_items)
pop_items

plt.plot(pop_items, 'ro')
plt.xlabel('Item index')
plt.ylabel('Number of interactions')
plt.show()

# # User activity

user_act = (URM_all > 0).sum(axis=1)
user_act = np.array(user_act).squeeze()
user_act = np.sort(user_act)
user_act

plt.plot(user_act, 'ro')
plt.xlabel('User index')
plt.ylabel('Number of interactions')
plt.show()

# # Interactions

interactions = URM_df[URM_df['data']>0]
interactions.min()

# All interactions are 1s. If there is some positive interaction, it is 1.

# # Check cold users

from src.data_management.data_reader import get_warm_user_rating_matrix

URM_all_warm = get_warm_user_rating_matrix(URM_all)

URM_all_warm

URM_all

# Some items and users are warm. For them, we have no ratings. Let's double check this.

user_act = (URM_all > 0).sum(axis=1)
user_act = np.array(user_act).squeeze()
user_act = np.sort(user_act)
condition = user_act == 0
cold_user = np.extract(condition, user_act)
cold_user.shape

pop_items = (URM_all > 0).sum(axis=0)
pop_items = np.array(pop_items).squeeze()
pop_items = np.sort(pop_items)
condition = pop_items == 0
cold_item = np.extract(condition, pop_items)
cold_item.shape

# There are around 3k cold items and users.

# # Discretization of user activity

# +
condition_list = []
threshold_list = [0, 1, 2, 5, 10, 50, 100, 1000]

t0 = threshold_list[0]
cond = user_act == threshold_list[0]
condition_list.append(cond)

t_prev = t0
for i in range(1, len(threshold_list)):
    cond = np.logical_and((user_act > t_prev), (user_act <= threshold_list[i]))
    condition_list.append(cond)
    t_prev = threshold_list[i]
    
# -

shape_list = []
for c in condition_list:
    temp = np.extract(c, user_act)
    shape_list.append(temp.shape[0])

shape_list

print("We have " + str(shape_list[0]) + " users with a " + str(threshold_list[0]) + " interactions")
for i in range (1, len(shape_list)):
    print("We have " + str(shape_list[i]) + " users with interactions in (" + str(threshold_list[i-1]) + ", " +str(threshold_list[i]) + "]")

# As we can see, most of very few users have more than 100 interactions. Half of the users have ratings in between 5 and 50. Let's visualize it with a graph.

shape_arr = np.array(shape_list)
total = shape_arr.sum()
normalized_shape = np.divide(shape_arr, total) 

normalized_shape

normalized_shape.sum()

# t_str = []
# for t in threshold_list:
#     t_str.append(str(t))
#
# plt.bar(t_str, normalized_shape, align='center', alpha=0.5)
#
# #plt.xticks(y_pos, objects)
# plt.ylabel('Usage')
# plt.title('Programming language usage')
#
# plt.show()

# User profiles seem to be a bit skewed in this sense. We expect the same things for the items

# # Discretization of item popularity

# +
condition_list = []
threshold_list = [0, 1, 2, 5, 10, 50, 100, 1000]

t0 = threshold_list[0]
cond = pop_items == threshold_list[0]
condition_list.append(cond)

t_prev = t0
for i in range(1, len(threshold_list)):
    cond = np.logical_and((pop_items > t_prev), (pop_items <= threshold_list[i]))
    condition_list.append(cond)
    t_prev = threshold_list[i]
    
# -

shape_list = []
for c in condition_list:
    temp = np.extract(c, pop_items)
    shape_list.append(temp.shape[0])

shape_list

print("We have " + str(shape_list[0]) + " items with a " + str(threshold_list[0]) + " interactions")
for i in range (1, len(shape_list)):
    print("We have " + str(shape_list[i]) + " items with interactions in (" + str(threshold_list[i-1]) + ", " +str(threshold_list[i]) + "]")

# +
shape_arr = np.array(shape_list)
total = shape_arr.sum()
normalized_shape = np.divide(shape_arr, total) 
t_str = []
for t in threshold_list:
    t_str.append(str(t))

plt.bar(t_str, normalized_shape, align='center', alpha=0.5)

#plt.xticks(y_pos, objects)
plt.ylabel('Item popularity')
plt.title('Threshold')

plt.show()
# -

# # Generalization of discretization
# Let's try to understand better the behavior generalizing this graph

t_list = [0, 1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 1000]
plot_popularity_discretized(user_act, t_list, y_label="Percentage of user activity")

# Most of the user have an activity level that is less then 20 interactions, and for many, we have only 1 interactions 

plot_popularity_discretized(pop_items, t_list, y_label="Item popularity")

# Some items are highly popular as we can seen, while this behaviour was not present for waht concerns the users.
# However, even in this case, we have a significant amount of items with ratings in (1, 10]

# # Joint analysis: user activity=1. What did they like?

user_act = (URM_all > 0).sum(axis=1)
user_act_unsorted = np.array(user_act).squeeze() 
user_act_unsorted

# +
user_who_gave_1_like = np.where(user_act_unsorted == 1, user_act_unsorted, 0)
index_list = []
for i in range(0, user_who_gave_1_like.size):
    if user_who_gave_1_like[i] == 1:
        index_list.append(i)
index_arr = np.array(index_list)

item_liked = []
for index in index_arr:
    item_liked.append(URM_all[index, :].indices[0])
item_liked_arr = np.array(item_liked)
item_liked_arr.size
# -

uniques = np.unique(item_liked_arr)
item_liked_arr.size - uniques.size

# There are 1143 items that received more than a like. We can try to understand if these items are popular or not

pop_items = get_popular_items(URM_all, popular_threshold=100)
pop_items.size

mask = np.in1d(item_liked_arr, pop_items)
pop_item_recommended = item_liked_arr[mask]
pop_item_recommended.size 

# #### Half of them are really popular! Let's see if some of them are unpopular.

unpop_items = get_unpopular_items(URM_all, 1)
mask = np.in1d(item_liked_arr, unpop_items)
unpop_item_recommended = item_liked_arr[mask]
unpop_item_recommended.size

# ##### 25 of them are due to the fact that only that user rated that item! They are somehow noise. 


