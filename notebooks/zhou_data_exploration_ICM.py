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

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# # EDA about ICM

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## Asset ICM

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_asset = pd.read_csv('../data/data_ICM_asset.csv')

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_asset.head()

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_asset.describe()

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_asset['data'].plot.hist(bins=20)

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_asset['data'].plot.box()

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
plt.plot(df_ICM_asset.sort_values(by='data', ascending=False)['data'].values)
plt.xlabel("Item random indices")
plt.ylabel("Asset value")
plt.show()

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Let's zoom in the histogram to see the distribution more or less

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
asset_values = df_ICM_asset['data']
asset_values[asset_values < asset_values.mean() + asset_values.std()*2].plot.hist(bins=20)

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Since the distribution does seem to be skewed, let's try to unskew it:

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
transformed_asset_values = np.log1p(1/asset_values)
plt.hist(transformed_asset_values, bins=20)
plt.xlabel("Asset")
plt.ylabel("Frequency")
plt.show()

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# This transformation does transform it into a gaussian, but what if the meaning of the asset is something
# about how important they are for the item, then by using similarity-based algorithm, it might be a bad
# thing.
#
# Regarding the missing values, we know that there are a total of 18495 items, then:

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
item_ids = df_ICM_asset['row'].unique()
mask = np.in1d(np.arange(0, 18495), item_ids, assume_unique=True)
item_missing_ids = np.arange(0, 18495)[~mask]
item_missing_ids

# + {"pycharm": {"name": "#%%\n"}}
missing_df_ICM_asset = pd.DataFrame(data={'row': item_missing_ids, 
                                          'col': np.repeat(0, len(item_missing_ids)), 
                                          'data': np.repeat(df_ICM_asset['data'].mean(), len(item_missing_ids))})
new_df_ICM_asset = pd.concat([df_ICM_asset, missing_df_ICM_asset], axis=0)
new_df_ICM_asset.describe()
# -

# Then, after adding this missing value, we can if necessary do the **unskewing**

# + {"heading_collapsed": true, "pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## Price 

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_price = pd.read_csv('../data/data_ICM_price.csv')

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_price.head()

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_price.describe()

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_price['data'].plot.hist(bins=20)

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_price['data'].plot.box()

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
plt.plot(df_ICM_price['data'].sort_values(ascending=False).values)
plt.xlabel("Item random indices")
plt.ylabel("Price")
plt.show()

# + {"hidden": true, "pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Let's zoom in the histogram of the price

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
price_values = df_ICM_price['data']
price_values[price_values < price_values.mean()+price_values.std()*2].plot.hist(bins=20)
plt.xlabel("")

# + {"hidden": true, "pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Let's also try to normalize it into a gaussian, but as we have said before, the meaning of this value is unknown, so,
# we do not know if this preprocessing is useful or not

# + {"hidden": true, "pycharm": {"is_executing": false, "name": "#%%\n"}}
transformed_price_values = np.log1p(1/price_values)
plt.hist(transformed_price_values, bins=20)
plt.ylabel("Frequency")
plt.xlabel("Price")
plt.show()

# + {"hidden": true, "cell_type": "markdown"}
# Let's also handle the missing values in the price ICM

# + {"hidden": true}
item_ids = df_ICM_price['row'].unique()
mask = np.in1d(np.arange(0, 18495), item_ids, assume_unique=True)
item_missing_ids = np.arange(0, 18495)[~mask]
item_missing_ids

# + {"hidden": true}
missing_df_ICM_price = pd.DataFrame(data={'row': item_missing_ids, 
                                          'col': np.repeat(0, len(item_missing_ids)), 
                                          'data': np.repeat(df_ICM_price['data'].mean(), len(item_missing_ids))})
new_df_ICM_price = pd.concat([df_ICM_price, missing_df_ICM_price], axis=0)
new_df_ICM_price.describe()

# + {"hidden": true, "cell_type": "markdown"}
# Then, after adding this missing value, we can if necessary do the **unskewing**

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# ## Sub class

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_sub_class = pd.read_csv('../data/data_ICM_sub_class.csv')

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_sub_class.head()

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_sub_class.describe()

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
print("There are %d unique items, which means there are all of them"%len(df_ICM_sub_class['row'].unique()))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Moreover, it means that each item has only one sub class

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
print("There are %d unique sub classes"%len(df_ICM_sub_class['col'].unique()))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Many items has the same sub class, then, let's see the top popular sub classes

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_sub_class['col'].value_counts()[:20].plot.bar()
plt.xlabel('Sub class')
plt.ylabel('Count')
plt.show()

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Let's also see the least populars

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
df_ICM_sub_class['col'].value_counts()[-20:].plot.bar()
plt.xlabel('Sub class')
plt.ylabel('Count')
plt.show()

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# Many sub classes are paired to only 1 items, let's see how many of them

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
counts = df_ICM_sub_class['col'].value_counts()
print("There are %d sub classes with only one items paired"%len(counts[counts==1]))

# + {"pycharm": {"name": "#%% md\n"}, "cell_type": "markdown"}
# We can also see a boxplot for more information

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}
counts.plot.box()

# + {"pycharm": {"is_executing": false, "name": "#%%\n"}}

