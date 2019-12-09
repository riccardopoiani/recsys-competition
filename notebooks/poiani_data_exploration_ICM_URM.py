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
from src.plots.plot_evaluation_helper import plot_popularity_discretized
from src.data_management.data_getter import *
# -

# # Data reading

URM_df = pd.read_csv('../data/data_train.csv')
ICM_asset_df = pd.read_csv('../data/data_ICM_asset.csv')
ICM_price_df = pd.read_csv('../data/data_ICM_price.csv')
ICM_subclass_df = pd.read_csv('../data/data_ICM_sub_class.csv') 

# +
user_list = URM_df['row']
item_list = URM_df['col']

user_list_unique = list(set(user_list))
item_list_unique = list(set(item_list))

URM_all = sps.coo_matrix((np.ones(len(URM_df)), (user_list, item_list)))
URM_all = URM_all.tocsr()
URM_all
# -

# # Price and URM

# ### Missing items
# Let's start from the item with the missing price and see their popularity in the URM.

item_ids = ICM_price_df['row'].unique()
mask = np.in1d(np.arange(0, 18495), item_ids, assume_unique=True)
item_missing_ids = np.arange(0, 18495)[~mask]
item_missing_ids

item_3513 = URM_df[URM_df['col'] == 3513]
item_3513.count()

item_10260 = URM_df[URM_df['col'] == 10260]
item_10260.count()

# The first item is somehow very unpopular. The second one has an almost average popularity. Indeed, consider the following.

# +
pop_items = (URM_all > 0).sum(axis=0)
pop_items = np.array(pop_items).squeeze()
pop_items = np.sort(pop_items)

threshold_list = [0, 16, 1000]
plot_popularity_discretized(pop_items, threshold_list, y_label="Percentage of element popularity")
# -

# ### Max price
# From the ICM exploration, we know that there is an item with max price.

max_price = ICM_price_df['data'].max()
item_max_price = ICM_price_df[ICM_price_df['data']==max_price]['row']
item_max_price = item_max_price.array[0]
item_max_price

high_price_item = URM_df[URM_df['row'] == item_max_price]
high_price_item.count()

# ### Price against popularity

item_popularity = URM_df['col'].value_counts()
pop_items = item_popularity[item_popularity >= item_popularity.quantile(q=0.9)].index
unpop_items = item_popularity[item_popularity <= item_popularity.quantile(q=0.1)].index

price_pop = ICM_price_df[ICM_price_df.row.isin(pop_items)]
price_unpop = ICM_price_df[ICM_price_df.row.isin(unpop_items)]

price_pop['data'].describe()

price_unpop['data'].describe()

ICM_price_df['data'].describe()

# There seems to be somwhow a difference in the price behaviour.
# Let's try to plot the price distribution for popular and unpopular items.

# +
price_pop_arr = price_pop['data']
price_pop_arr = np.array(price_pop_arr.array)

price_unpop_arr = price_unpop['data']
price_unpop_arr = np.array(price_unpop_arr)

price_pop_arr.sort()
price_unpop_arr.sort()
# -

plt.title('Histogram of prices of popular items')
plt.xlabel('Price')
plt.ylabel('Number of popular items')
plt.hist(price_pop_arr, bins = 'auto')
plt.show()

plt.title('Histogram of prices of unpopular items')
plt.xlabel('Price')
plt.ylabel('Number of unpopular items')
plt.hist(price_unpop_arr, bins = 'auto')
plt.show()

# Distributions look similar, however, scales are different. Let's try to zoom better on items with a lower price.

# +
price_pop_low_price = price_pop[price_pop['data'] < 0.025]
price_unpop_low_price = price_unpop[price_unpop['data'] < 0.025]

price_pop_arr_low_price = price_pop_low_price['data']
price_pop_arr_low_price = np.array(price_pop_arr_low_price.array)

price_unpop_arr_low_price = price_unpop_low_price['data']
price_unpop_arr_low_price = np.array(price_unpop_arr_low_price)

price_pop_arr_low_price.sort()
price_unpop_arr_low_price.sort()
# -

plt.title('Histogram of prices of popular items with low price')
plt.xlabel('Price')
plt.ylabel('Number of popular items')
plt.hist(price_pop_arr_low_price, bins = 'auto')
plt.show()

plt.title('Histogram of prices of unpopular items with low price')
plt.xlabel('Price')
plt.ylabel('Number of unpopular items')
plt.hist(price_unpop_arr_low_price, bins = 'auto')
plt.show()

# Popular items are more skewed toward lower prices. While the unpopular decreasing trend is smoother.

# # Asset and URM
#
# ### Missing items

item_ids = ICM_asset_df['row'].unique()
mask = np.in1d(np.arange(0, 18495), item_ids, assume_unique=True)
item_missing_ids = np.arange(0, 18495)[~mask]
item_missing_ids

for item in item_missing_ids:
    elem = URM_df[URM_df['col'] == item]
    print("For item " + str(item) + ", we have a popularity of " + str(elem.count()['row']))

# ### High asset

max_price = ICM_asset_df['data'].max()
item_max_price = ICM_asset_df[ICM_asset_df['data']==max_price]['row']
item_max_price = item_max_price.array[0]
item_max_price

# It is the same of before! (popularity = 5)

# ### Asset against popularity

# +
item_popularity = URM_df['col'].value_counts()
pop_items = item_popularity[item_popularity >= item_popularity.quantile(q=0.9)].index
unpop_items = item_popularity[item_popularity <= item_popularity.quantile(q=0.1)].index

asset_pop = ICM_asset_df[ICM_asset_df.row.isin(pop_items)]
asset_unpop = ICM_asset_df[ICM_asset_df.row.isin(unpop_items)]
# -

asset_pop.describe()

asset_unpop.describe()

ICM_asset_df.describe()

# +
asset_pop_arr = asset_pop['data']
asset_pop_arr = np.array(asset_pop_arr.array)

asset_unpop_arr = asset_unpop['data']
asset_unpop_arr = np.array(asset_unpop_arr)

asset_pop_arr.sort()
asset_unpop_arr.sort()
# -

plt.title('Histogram of assets of popular items')
plt.xlabel('Asset')
plt.ylabel('Number of popular items')
plt.hist(asset_pop_arr, bins = 'auto')
plt.show()

plt.title('Histogram of assets of unpopular items')
plt.xlabel('Asset')
plt.ylabel('Number of unpopular items')
plt.hist(asset_unpop_arr, bins = 'auto')
plt.show()

# +
asset_pop_low_asset = asset_pop[asset_pop['data'] < 0.02]
asset_unpop_low_asset = asset_unpop[asset_unpop['data'] < 0.02]

asset_pop_arr_low_asset = asset_pop_low_asset['data']
asset_pop_arr_low_asset = np.array(asset_pop_arr_low_asset.array)

asset_unpop_arr_low_asset = asset_unpop_low_asset['data']
asset_unpop_arr_low_asset = np.array(asset_unpop_arr_low_asset)

asset_pop_arr_low_asset.sort()
asset_unpop_arr_low_asset.sort()
# -

plt.title('Histogram of assets of popular items with low aset')
plt.xlabel('Asset')
plt.ylabel('Number of popular items')
plt.hist(asset_pop_arr_low_asset, bins = 'auto')
plt.show()

plt.title('Histogram of assets of unpopular items with low aset')
plt.xlabel('Asset')
plt.ylabel('Number of unpopular items')
plt.hist(asset_unpop_arr_low_asset, bins = 'auto')
plt.show()

# Same conclusion of before

# # Subclass and URM

ICM_subclass_df['col'].value_counts()[:20].plot.bar()
plt.xlabel('Sub class')
plt.ylabel('Count')
plt.show()

# ### Popularity against subclass

subclass_arr = ICM_subclass_df['col'].unique()

pop_class_dict = {}
pop_class_list = []
subclass_count_list = []
subclass_count_dict = {}
for elem in subclass_arr:
    # Popularity of the class
    item_class = ICM_subclass_df[ICM_subclass_df['col'] == elem]['row'] # items of class elem
    item_class = np.array(item_class.array) # item of class elem nparray version
    pop_class = URM_df[URM_df['col'].isin(item_class)].count()['row']
    
    # Count of the subclass
    subclass_count_list.append(item_class.size)
    subclass_count_dict[elem] = item_class.size
    
    pop_class_list.append(pop_class)
    pop_class_dict[elem] = pop_class

ICM_subclass_df['col'].value_counts()[:20].min()

count_tick = []
pop_tick = []
key_tick = []
for key in subclass_count_dict.keys():
    if subclass_count_dict[key] >= 75:
        count_tick.append(subclass_count_dict[key])
        pop_tick.append(pop_class_dict[key])
        key_tick.append(key)

result = pd.DataFrame(
    {'count': count_tick,
     'popularity': pop_tick,
     'subclass': key_tick
    })

result['popularity'].plot.bar()
plt.title("Popularity of most common subclasses")
plt.ylabel("Popularity")
plt.xlabel("Subclass sorted in decreasing order of appearence")

count_tick = []
pop_tick = []
key_tick = []
for key in subclass_count_dict.keys():
    if subclass_count_dict[key] == 1:
        count_tick.append(subclass_count_dict[key])
        pop_tick.append(pop_class_dict[key])
        key_tick.append(key)

result = pd.DataFrame(
    {'count': count_tick,
     'popularity': pop_tick,
     'subclass': key_tick
    })

result['popularity'][:20].plot.bar()
plt.title("Popularity of least common subclasses")
plt.ylabel("Popularity")
plt.xlabel("Subclass sorted in decreasing order of appearence")

result['popularity'].plot.bar()
plt.title("Popularity of least common subclasses")
plt.ylabel("Popularity")
plt.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom=False,      # ticks along the bottom edge are off
    top=False,         # ticks along the top edge are off
    labelbottom=False) # labels along the bottom edge are off
plt.xlabel("Subclass sorted in decreasing order of appearence")


