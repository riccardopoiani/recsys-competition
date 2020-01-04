import time

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from tqdm import tqdm

from course_lib.Base.BaseRecommender import BaseRecommender
from src.data_management.data_preprocessing_fm import sample_negative_interactions_uniformly
from src.utils.general_utility_functions import get_total_number_of_users, get_total_number_of_items
from sklearn.preprocessing import MinMaxScaler


def preprocess_dataframe_after_reading(df: pd.DataFrame):
    df = df.copy()
    df = df.sort_values(by="user_id", ascending=True)
    df = df.reset_index()
    df = df.drop(columns=["index"], inplace=False)
    return df


def get_valid_dataframe_second_version(user_id_array, cutoff, main_recommender, path, mapper, recommender_list,
                                       URM_train, user_factors=None, item_factors=None):
    data_frame = get_boosting_base_dataframe(user_id_array=user_id_array, top_recommender=main_recommender,
                                             exclude_seen=True, cutoff=cutoff)
    for rec in recommender_list:
        data_frame = add_recommender_predictions(data_frame=data_frame, recommender=rec,
                                                 column_name=rec.RECOMMENDER_NAME)

    data_frame = advanced_subclass_handling(data_frame=data_frame, URM_train=URM_train, path=path)
    data_frame = add_ICM_information(data_frame=data_frame, path=path, one_hot_encoding_subclass=False,
                                     use_subclass=True)
    data_frame = add_UCM_information(data_frame=data_frame, path=path, user_mapper=mapper)
    data_frame = add_user_len_information(data_frame=data_frame, URM_train=URM_train)
    data_frame = add_item_popularity(data_frame=data_frame, URM_train=URM_train)
    if user_factors is not None:
        data_frame = add_user_factors(data_frame=data_frame, user_factors=user_factors)
    if item_factors is not None:
        data_frame = add_item_factors(data_frame=data_frame, item_factors=item_factors)

    data_frame = data_frame.sort_values(by="user_id", ascending=True)
    data_frame = data_frame.reset_index()
    data_frame = data_frame.drop(columns=["index"], inplace=False)

    return data_frame


def get_train_dataframe_proportion(user_id_array, cutoff, main_recommender, path, mapper, recommender_list,
                                   URM_train, proportion, user_factors=None, item_factors=None,
                                   negative_label_value=0, threshold=0.7):
    data_frame = get_boosting_base_dataframe(user_id_array=user_id_array, top_recommender=main_recommender,
                                             exclude_seen=False, cutoff=cutoff)
    labels, non_zero_count, _ = get_label_array(data_frame, URM_train)
    data_frame['label'] = labels
    data_frame = add_random_negative_ratings(data_frame=data_frame, URM_train=URM_train, proportion=proportion,
                                             negative_label_value=negative_label_value)

    data_frame = data_frame.sort_values(by="user_id", ascending=True)
    data_frame = data_frame.reset_index()
    data_frame = data_frame.drop(columns=["index"], inplace=False)

    for rec in recommender_list:
        data_frame = add_recommender_predictions(data_frame=data_frame, recommender=rec,
                                                 column_name=rec.RECOMMENDER_NAME)
        # Add labels value in order to differentiate more the elements
        mask = (data_frame[rec.RECOMMENDER_NAME] > threshold) & (data_frame['label'] > 0)
        print("\t Score greater than threshold: {}/{}".format(np.sum(mask), non_zero_count))
        data_frame.loc[mask, 'label'] += 1
    print("Labels greater than 1: {}".format(np.sum(data_frame['label'] > 1)))

    data_frame = advanced_subclass_handling(data_frame=data_frame, URM_train=URM_train, path=path, add_subclass=False)
    data_frame = add_ICM_information(data_frame=data_frame, path=path, one_hot_encoding_subclass=False,
                                     use_subclass=True)
    data_frame = add_UCM_information(data_frame=data_frame, path=path, user_mapper=mapper)
    data_frame = add_user_len_information(data_frame=data_frame, URM_train=URM_train)
    data_frame = add_item_popularity(data_frame=data_frame, URM_train=URM_train)
    if user_factors is not None:
        data_frame = add_user_factors(data_frame=data_frame, user_factors=user_factors)
    if item_factors is not None:
        data_frame = add_item_factors(data_frame=data_frame, item_factors=item_factors)

    data_frame = data_frame.sort_values(by="user_id", ascending=True)
    data_frame = data_frame.reset_index()
    data_frame = data_frame.drop(columns=["index"], inplace=False)

    return data_frame


def get_dataframe_all_data(user_id_array, path, mapper, recommender_list,
                           URM_train, proportion, user_factors=None, item_factors=None):
    negative_URM = sample_negative_interactions_uniformly(negative_sample_size=len(URM_train.data) * proportion,
                                                          URM=URM_train)
    data_frame = get_dataframe_URM(user_id_array=user_id_array, URM_train=URM_train + negative_URM)
    labels, _, _ = get_label_array(data_frame, URM_train)
    data_frame['label'] = labels

    data_frame = data_frame.sort_values(by="user_id", ascending=True)
    data_frame = data_frame.reset_index()
    data_frame = data_frame.drop(columns=["index"], inplace=False)

    for rec in recommender_list:
        data_frame = add_recommender_predictions(data_frame=data_frame, recommender=rec,
                                                 column_name=rec.RECOMMENDER_NAME)

    data_frame = advanced_subclass_handling(data_frame=data_frame, URM_train=URM_train, path=path, add_subclass=False)
    data_frame = add_ICM_information(data_frame=data_frame, path=path, one_hot_encoding_subclass=False,
                                     use_subclass=True)
    data_frame = add_UCM_information(data_frame=data_frame, path=path, user_mapper=mapper)
    data_frame = add_user_len_information(data_frame=data_frame, URM_train=URM_train)
    data_frame = add_item_popularity(data_frame=data_frame, URM_train=URM_train)
    if user_factors is not None:
        data_frame = add_user_factors(data_frame=data_frame, user_factors=user_factors)
    if item_factors is not None:
        data_frame = add_item_factors(data_frame=data_frame, item_factors=item_factors)

    data_frame = data_frame.sort_values(by="user_id", ascending=True)
    data_frame = data_frame.reset_index()
    data_frame = data_frame.drop(columns=["index"], inplace=False)

    return data_frame


def get_dataframe_first_version(user_id_array, remove_seen_flag, cutoff, main_recommender, path, mapper,
                                recommender_list,
                                URM_train):
    # Get dataframe for these users
    data_frame = get_boosting_base_dataframe(user_id_array=user_id_array, exclude_seen=remove_seen_flag,
                                             cutoff=cutoff, top_recommender=main_recommender)
    for rec in recommender_list:
        data_frame = add_recommender_predictions(data_frame=data_frame, recommender=rec,
                                                 column_name=rec.RECOMMENDER_NAME)

    data_frame = add_ICM_information(data_frame=data_frame, path=path)
    data_frame = add_UCM_information(data_frame=data_frame, path=path, user_mapper=mapper)
    data_frame = add_user_len_information(data_frame=data_frame, URM_train=URM_train)

    data_frame = data_frame.sort_values(by="user_id", ascending=True)
    data_frame = data_frame.reset_index()
    data_frame.drop(columns=["index"], inplace=False)

    return data_frame


def add_user_factors(data_frame: pd.DataFrame, user_factors: np.ndarray):
    """
    Add user factors to the dataframe

    :param data_frame:
    :param user_factors:
    :return:
    """
    print("Adding user factors...")

    data_frame = data_frame.copy()
    user_factors_df = pd.DataFrame(data=user_factors,
                                   index=np.arange(0, user_factors.shape[0]),
                                   columns=["user_factor_{}".format(i + 1) for i in range(user_factors.shape[1])])
    data_frame = pd.merge(data_frame, user_factors_df, left_on="user_id", right_index=True)

    return data_frame


def add_item_factors(data_frame: pd.DataFrame, item_factors: np.ndarray):
    """
    Add item factors to the dataframe

    :param data_frame:
    :param item_factors:
    :return:
    """
    print("Adding item factors...")

    data_frame = data_frame.copy()
    item_factors_df = pd.DataFrame(data=item_factors,
                                   index=np.arange(0, item_factors.shape[0]),
                                   columns=["item_factor_{}".format(i + 1) for i in range(item_factors.shape[1])])
    data_frame = pd.merge(data_frame, item_factors_df, left_on="item_id", right_index=True)

    return data_frame


def add_item_popularity(data_frame: pd.DataFrame, URM_train: csr_matrix):
    """
    Add the item popularity to the dataframe

    :param data_frame: data frame containing information for boosting
    :param URM_train: URM train matrix
    :return: dataframe containing boosting information + item popularity
    """
    print("Adding item popularity...")
    data_frame = data_frame.copy()

    pop_items = (URM_train > 0).sum(axis=0)
    pop_items = np.array(pop_items).squeeze()
    item_ids = np.arange(URM_train.shape[1])
    data = np.array([item_ids, pop_items])
    data = np.transpose(data)

    new_df = pd.DataFrame(data=data, columns=["row", "item_pop"])

    data_frame = pd.merge(data_frame, new_df, left_on="item_id", right_on="row")
    data_frame = data_frame.drop(columns=["row"], inplace=False)

    return data_frame


def get_label_array(data_frame: pd.DataFrame, URM_train: csr_matrix):
    """
    Create a dataframe with a single column with the correct predictions

    :param data_frame: data frame containing information for boosting
    :param URM_train: URM train matrix
    :return: numpy array containing y information
    """
    print("Retrieving training labels...")
    user_ids = data_frame['user_id'].values
    item_ids = data_frame['item_id'].values

    y = np.zeros(user_ids.size, dtype=np.int)
    labels = np.array(URM_train[user_ids, item_ids].tolist()).flatten()
    y[labels > 0] = 1

    non_zero_count = np.count_nonzero(y)
    print("\t- There are {} non-zero ratings in {}".format(non_zero_count, y.size))

    return y, non_zero_count, y.size


def add_user_len_information(data_frame: pd.DataFrame, URM_train: csr_matrix):
    """
    Add information concerning the user profile length to the row of the dataframe

    :param data_frame: data frame that is being pre-processed from boosting
    :param URM_train: URM train from which to take profile length information
    :return: data frame with new content inserted
    """
    print("Adding user profile length...")
    data_frame = data_frame.copy()

    user_act = (URM_train > 0).sum(axis=1)
    user_act = np.array(user_act).squeeze()
    user_ids = np.arange(URM_train.shape[0])
    data = np.array([user_ids, user_act])
    data = np.transpose(data)
    new_df = pd.DataFrame(data=data, columns=["row", "user_act"])

    data_frame = pd.merge(data_frame, new_df, left_on="user_id", right_on="row")
    data_frame = data_frame.drop(columns=["row"], inplace=False)

    return data_frame


def remap_data_frame(df: pd.DataFrame, mapper):
    """
    Change user_id columns of the df given in input, according to the mapper.
    Users that are not present will be removed, and the others will be mapped to the correct number.


    :param df: dataframe that will be modified
    :param mapper: mapper according to which the dataframe will be modified
    :return: dataframe with "user_id" column modified properly
    """
    df = df.copy()

    # Remove users that are not present in the mapper
    original_users = df['row'].values
    new_users_key = list(mapper.keys())
    new_users_key = list(map(int, new_users_key))
    new_users_key = np.array(new_users_key)
    mask = np.in1d(original_users, new_users_key, invert=True)
    remove = original_users[mask]
    df = df.set_index("row")
    mask = np.in1d(df.index, remove)
    df = df.drop(df.index[mask])

    # Map the index to the new one
    df = df.reset_index()
    df['row'] = df['row'].map(lambda x: mapper[str(x)])

    return df


def add_UCM_information(data_frame: pd.DataFrame, user_mapper, path="../../data/", use_region=True, use_age=True,
                        use_age_onehot=False):
    """
    Add UCM information to the data frame for XGboost

    :param data_frame: data frame containing information being pre-processed for boosting
    :param user_mapper: mapper original users to train users
    :param path: where to read UCM csv files
    :param use_region: True is region information should be used, false otherwise
    :param use_age: True if age information should be used, false otherwise
    :param use_age_onehot: True if age information added is one hot, false otherwise
    :return: pd.DataFrame containing the original data frame+ UCM information
    """
    print("Adding UCM information...")
    t_users = get_total_number_of_users()  # Total number of users (-1 since indexing from 0)

    data_frame = data_frame.copy()
    df_region: pd.DataFrame = pd.read_csv(path + "data_UCM_region.csv")
    df_age: pd.DataFrame = pd.read_csv(path + "data_UCM_age.csv")

    # Re-map UCM data frame in order to have the correct user information
    if use_region:
        df_region = df_region[['row', 'col']]

        df_dummies = pd.get_dummies(df_region['col'], prefix='region')
        df_dummies = df_dummies.join(df_region['row'])
        df_dummies = df_dummies.groupby(['row'], as_index=False).sum()

        # Fill missing values
        user_present = df_dummies['row'].values
        total_users = np.arange(t_users)
        mask = np.in1d(total_users, user_present, invert=True)
        missing_users = total_users[mask]
        num_col = df_dummies.columns.size
        imputed_users = np.zeros(shape=(num_col, missing_users.size))
        imputed_users[0] = missing_users
        missing_df = pd.DataFrame(data=np.transpose(imputed_users), dtype=np.int32, columns=df_dummies.columns)
        df_region_onehot = df_dummies.append(missing_df, sort=False)

        if user_mapper is not None:
            df_region_onehot = remap_data_frame(df=df_region_onehot, mapper=user_mapper)
        data_frame = pd.merge(data_frame, df_region_onehot, right_on="row", left_on="user_id")
        data_frame = data_frame.drop(columns=["row"], inplace=False)

    if use_age:
        df_age = df_age[['row', 'col']]

        # Handle missing values: fill with mode + 1
        users_present = df_age['row'].values
        total_users = np.arange(t_users)
        mask = np.in1d(total_users, users_present, invert=True)
        missing_users = total_users[mask].astype(np.int32)
        missing_val_filled = np.ones(missing_users.size) * (int(df_age['col'].mode()) + 1)
        missing = np.array([missing_users, missing_val_filled], dtype=np.int32)
        missing_df = pd.DataFrame(data=np.transpose(missing), columns=["row", "col"])
        df_age_imputed = df_age.copy().append(missing_df, sort=False)
        df_age_imputed = df_age_imputed.reset_index()
        df_age_imputed = df_age_imputed[['row', 'col']]

        if user_mapper is not None:
            df_age_imputed = remap_data_frame(df=df_age_imputed, mapper=user_mapper)
        df_age_imputed = df_age_imputed.rename(columns={"col": "age"})
        if use_age_onehot:
            row = df_age_imputed['row']
            df_age_imputed = pd.get_dummies(df_age_imputed['age'], prefix='age')
            df_age_imputed = df_age_imputed.join(row)

        data_frame = pd.merge(data_frame, df_age_imputed, right_on="row", left_on="user_id")
        data_frame = data_frame.drop(columns=["row"], inplace=False)

        # Add dummy variables indicating that the region has been imputed
        df_age_dummy_imputation = df_age.copy()
        df_age_dummy_imputation['col'] = 0
        imputed_df = pd.DataFrame(
            data={"row": missing_users, "col": np.ones(shape=missing_users.size, dtype=np.int)})
        df_age_dummy_imputation = df_age_dummy_imputation.append(imputed_df, sort=False)
        df_age_dummy_imputation = df_age_dummy_imputation.rename(columns={"col": "age_imputed_flag"})
        if user_mapper is not None:
            df_age_dummy_imputation = remap_data_frame(df=df_age_dummy_imputation, mapper=user_mapper)
        data_frame = pd.merge(data_frame, df_age_dummy_imputation, right_on="row", left_on="user_id")
        data_frame = data_frame.drop(columns=["row"], inplace=False)

    return data_frame


def advanced_subclass_handling(data_frame: pd.DataFrame, URM_train: csr_matrix, path="../../data/",
                               add_subclass=False):
    """
    Here we want to include in the training set sub class information in the following way:
    - A column encoding the mean of 'label' for a certain couple (user, subclass): i.e. how many
    items of that subclass the user liked
    - Including information about the popularity of the subclass (how many items for that subclass
    - Including ratings of that subclass

    :param URM_train: mean response will be retrieved from here
    :param data_frame: dataframe being pre-processed for boosting
    :param path: path to the folder containing subclass dataframe
    :return: dataframe with augmented information
    """
    print("Adding subclass and feature engineering subclass...")
    data_frame = data_frame.copy()

    df_subclass: pd.DataFrame = pd.read_csv(path + "data_ICM_sub_class.csv")
    df_subclass = df_subclass[['row', 'col']]
    df_subclass = df_subclass.rename(columns={"col": "subclass"})

    # Merging sub class information
    data_frame = pd.merge(data_frame, df_subclass, right_on="row", left_on="item_id")
    data_frame = data_frame.drop(columns=["row"], inplace=False)

    print("\t- Add items present for each subclass")
    # Add subclass item-popularity: how many items are present of that subclass
    subclass_item_count = df_subclass.groupby("subclass").count()
    data_frame = pd.merge(data_frame, subclass_item_count, right_index=True, left_on="subclass")
    data_frame = data_frame.rename(columns={"row": "item_per_subclass"})

    print("\t- Add ratings popularity for each subclass")
    # Add subclass ratings-popularity: how many interactions we have for each subclass
    URM_train_csc = URM_train.tocsc()
    n_ratings_sub = []

    sorted_sub_indices = np.argsort(df_subclass['subclass'].values)
    sorted_sub = df_subclass['subclass'][sorted_sub_indices].values
    sorted_item_subclass = df_subclass['row'][sorted_sub_indices].values

    unique_sorted_sub, sub_indptr = np.unique(sorted_sub, return_index=True)
    sub_indptr = np.concatenate([sub_indptr, [sorted_sub.size]])
    for i, sub in tqdm(enumerate(unique_sorted_sub), total=unique_sorted_sub.size, desc="\t\tProcessing"):
        item_sub = sorted_item_subclass[sub_indptr[i]: sub_indptr[i + 1]]
        n_ratings_sub.append(URM_train_csc[:, item_sub].data.size)

    ratings_sub = np.array([unique_sorted_sub, n_ratings_sub])
    ratings_per_sub_df = pd.DataFrame(data=np.transpose(ratings_sub),
                                      columns=["subclass", "global_ratings_per_subclass"])

    data_frame = pd.merge(data_frame, ratings_per_sub_df, left_on="subclass", right_on="subclass")

    # Add subclass ratings-popularity for each user using rating percentage
    print("\t- Add ratings popularity for pairs (user, subclass)")
    users = data_frame['user_id'].values
    sub = data_frame['subclass'].values

    perc_array = np.zeros(users.size)
    rat_array = np.zeros(users.size)
    for i, user in tqdm(enumerate(users), total=users.size, desc="\t\tProcessing"):
        curr_sub = sub[i]
        curr_sub_index = np.searchsorted(unique_sorted_sub, curr_sub)

        # Find items of this subclass
        item_sub = sorted_item_subclass[sub_indptr[curr_sub_index]: sub_indptr[curr_sub_index + 1]]
        user_item = URM_train.indices[URM_train.indptr[user]: URM_train.indptr[user + 1]]

        total_user_likes = user_item.size
        mask = np.in1d(item_sub, user_item)
        likes_per_sub = item_sub[mask].size
        user_p = likes_per_sub / total_user_likes
        perc_array[i] = user_p
        rat_array[i] = likes_per_sub

    data_frame["subclass_user_like_perc"] = perc_array
    data_frame["subclass_user_like_quantity"] = rat_array

    if not add_subclass:
        data_frame = data_frame.drop(columns=["subclass"], inplace=False)

    return data_frame


def add_ICM_information(data_frame: pd.DataFrame, path="../../data/", use_price=True, use_asset=True,
                        use_subclass=True, one_hot_encoding_subclass=False):
    """
    Add information form the ICM files to the data frame

    :param one_hot_encoding_subclass: if one hot encoding should be applied to subclass or not
    :param data_frame: data frame that is being pre-processed for boosting
    :param path: path to the folder containing the csv files
    :param use_price: True if you wish to append price information, false otherwise
    :param use_asset: True if you wish to append asset information, false otherwise
    :param use_subclass: True if you wish to append subclass information, false otherwise
    :return: pd.DataFrame containing the information
    """
    print("Adding ICM information...")
    data_frame = data_frame.copy()
    df_price: pd.DataFrame = pd.read_csv(path + "data_ICM_price.csv")
    df_asset: pd.DataFrame = pd.read_csv(path + "data_ICM_asset.csv")
    df_subclass: pd.DataFrame = pd.read_csv(path + "data_ICM_sub_class.csv")

    total_items = get_total_number_of_items()
    total_items = np.arange(total_items)

    if use_price:
        # Handle missing values
        item_present = df_price['row'].values
        mask = np.in1d(total_items, item_present, invert=True)
        missing_items = total_items[mask].astype(np.int32)
        missing_val_filled = np.ones(missing_items.size) * df_price['data'].median()
        missing = np.array([missing_items, missing_val_filled])
        missing_df = pd.DataFrame(data=np.transpose(missing), columns=['row', 'data'])
        df_price = df_price.append(missing_df, sort=False)
        df_price = df_price.reset_index()
        df_price = df_price[['row', 'data']]

        # TODO remove outliers and add dummy variable

        df_price = df_price.rename(columns={"data": "price"})
        data_frame = pd.merge(data_frame, df_price, right_on="row", left_on="item_id")
        data_frame = data_frame.drop(columns=['row'], inplace=False)
    if use_asset:
        # Handle missing values
        item_present = df_asset['row'].values
        mask = np.in1d(total_items, item_present, invert=True)
        missing_items = total_items[mask].astype(np.int32)
        missing_val_filled = np.ones(missing_items.size) * df_asset['data'].median()
        missing = np.array([missing_items, missing_val_filled])
        missing_df = pd.DataFrame(data=np.transpose(missing), columns=['row', 'data'])
        df_asset = df_asset.append(missing_df, sort=False)
        df_asset = df_asset.reset_index()
        df_asset = df_asset[['row', 'data']]

        # TODO remove outliers and add dummy variable

        df_asset = df_asset.rename(columns={"data": "asset"})
        data_frame = pd.merge(data_frame, df_asset, right_on="row", left_on="item_id")
        data_frame = data_frame.drop(columns=["row"], inplace=False)
    if use_subclass:
        df_subclass = df_subclass[['row', 'col']]
        df_subclass = df_subclass.rename(columns={"col": "subclass"})

        if not one_hot_encoding_subclass:
            data_frame = pd.merge(data_frame, df_subclass, right_on="row", left_on="item_id")
        else:
            dummies = pd.get_dummies(df_subclass['subclass'])
            dummies = dummies.join(df_subclass['row'])
            data_frame = pd.merge(data_frame, dummies, right_on="row", left_on="item_id")

        data_frame = data_frame.drop(columns=["row"], inplace=False)

    return data_frame


def add_recommender_predictions(data_frame: pd.DataFrame, recommender: BaseRecommender,
                                column_name: str, min_max_scaling=True):
    """
    Add predictions of a recommender to the dataframe and return the new dataframe

    Note: Assumes that the data_frame is ordered by user_id (increasingly)

    :param data_frame: dataframe on which predictions will be added
    :param recommender: recommender of which the predictions will be added
    :param column_name: name of the new column
    :param min_max_scaling: whether to apply min-max scaling or not
    :return: new dataframe containing recommender predictions
    """
    print("Adding recommender predictions - COLUMN NAME: {}".format(column_name))

    new_df = data_frame.copy()
    items = new_df['item_id'].values.astype(int)
    users = new_df['user_id'].values.astype(int)

    # Check if dataframe is sorted by user_id
    if not np.all(users[i] <= users[i + 1] for i in range(users.size - 1)):
        raise ValueError("The dataframe is not sorted by user_id")

    prediction_values = np.zeros(items.size, dtype=np.float32)

    # Use indptr to avoid using query of dataframe
    unique_users, user_indptr = np.unique(users, return_index=True)
    user_indptr = np.concatenate([user_indptr, [users.size]])
    all_scores = recommender._compute_item_score(unique_users)

    if min_max_scaling:
        scaler = MinMaxScaler()
        scaler.fit(all_scores.reshape(-1, 1))
        all_scores = np.reshape(scaler.transform(all_scores.reshape(-1, 1)), newshape=all_scores.shape)

    for i, user_id in tqdm(enumerate(unique_users), total=unique_users.size,
                           desc="\tAdd users predictions".format(column_name)):
        items_for_user_id = items[user_indptr[i]: user_indptr[i + 1]]
        scores = all_scores[i, items_for_user_id].copy()
        prediction_values[user_indptr[i]: user_indptr[i + 1]] = scores

    new_df[column_name] = prediction_values

    del all_scores  # Remove this variable in order to let the garbage collector collect it

    return new_df


def user_uniform_sampling(user: int, URM_train: csr_matrix, items_to_exclude: np.array, sample_size: int,
                          batch_size=1000):
    """
    Sample negative interactions at random for a given users from URM_train

    :param items_to_exclude: exclude these items from the sampling
    :param user: sample negative interactions for this user
    :param URM_train: URM from which samples will be taken
    :param sample_size: how many samples to take
    :param batch_size: batch size dimension for the number of random sampling to do at each iteration
    :return: np.array containing the collected samples
    """

    sampled = 0

    invalid_items = URM_train.indices[URM_train.indptr[user]: URM_train.indptr[user + 1]]
    collected_samples = []

    while sampled < sample_size:
        items_sampled = np.random.randint(low=0, high=URM_train.shape[1], size=batch_size)
        items_sampled = np.unique(items_sampled)

        # Remove items already seen and items to exclude
        valid_items = np.setdiff1d(items_sampled, invalid_items, assume_unique=True)
        valid_items = np.setdiff1d(valid_items, items_to_exclude, assume_unique=True)

        # Cap the size of batch size if it is the last batch
        if sampled + len(valid_items) > sample_size:
            remaining_sample_size = sample_size - sampled
            valid_items = valid_items[:remaining_sample_size]
        collected_samples = np.concatenate([collected_samples, valid_items])

        # Update invalid items
        invalid_items = np.concatenate([invalid_items, valid_items])
        sampled += len(valid_items)

    return collected_samples


def add_random_negative_ratings(data_frame: pd.DataFrame, URM_train: csr_matrix, proportion=1, negative_label_value=0):
    """
    Add random negative rating sampled from URM train

    Note: labels should be already inserted in the dataframe for this purpose in a 'label' column
    Note: Assumes that the dataframe is ordered based on the users

    :param URM_train: URM train from which negative ratings will be sampled
    :param data_frame: dataframe on which these negative ratings will be added
    :param proportion: proportion w.r.t. the positive ratings (expressed as positive/negative)
    :param negative_label_value: the label to assign for negative sampled ratings
    :return: a new dataframe containing more negative interactions
    """
    data_frame = data_frame.copy()

    label_list = data_frame['label'].values.astype(int)
    item_list = data_frame['item_id'].values.astype(int)
    users, user_indptr = np.unique(data_frame['user_id'].values.astype(int), return_index=True)
    user_indptr = np.concatenate([user_indptr, [data_frame['user_id'].size]])

    new_user_list = []
    new_item_list = []

    for i, user in tqdm(enumerate(users), desc="\tSample negative ratings", total=users.size):
        pos_labels = label_list[user_indptr[i]: user_indptr[i + 1]]
        pos_count = np.count_nonzero(pos_labels)
        total = pos_labels.size
        neg_count = total - pos_count
        samples_to_add = np.array([int(pos_count / proportion) - neg_count]).min()
        if samples_to_add > 0:
            items_to_exclude = item_list[user_indptr[i]: user_indptr[i + 1]]
            samples = user_uniform_sampling(user, URM_train, sample_size=samples_to_add,
                                            items_to_exclude=items_to_exclude)
            new_user_list.extend([user] * samples_to_add)
            new_item_list.extend(samples.tolist())

    data = np.array([new_user_list, new_item_list])
    new_df = pd.DataFrame(np.transpose(data), columns=['user_id', 'item_id'], dtype=np.int)
    new_df['label'] = negative_label_value

    new_df = data_frame.append(new_df, sort=False)
    return new_df


def get_dataframe_URM(user_id_array: list, URM_train: csr_matrix):
    URM_train_slice = URM_train[user_id_array, :]
    data_frame = pd.DataFrame({"user_id": URM_train_slice.tocoo().row, "item_id": URM_train_slice.tocoo().col},
                              dtype=np.int)
    return data_frame


def get_boosting_base_dataframe(user_id_array, top_recommender: BaseRecommender, cutoff: int,
                                exclude_seen=False):
    """
    Get boosting data-frame preprocessed.
    In particular, it will create a data frame containing "user_id" presents in user_id_array, and computing
    possible recommendations using top_recommender, from which "item_id" will be extracted

    :param exclude_seen:
    :param user_id_array: user that you want to recommend
    :param top_recommender: top recommender used for building the dataframe
    :param cutoff: if you are interested in MAP@10, choose a large number, for instance, 20
    :return: dataframe containing the described information
    """
    print("Retrieving base dataframe using the main recommender...")
    # Setting items
    if exclude_seen:
        recommendations = np.array(top_recommender.recommend(user_id_array=user_id_array, cutoff=cutoff,
                                                             remove_seen_flag=exclude_seen))
    else:
        # Generate recommendations
        double_rec_false = np.array(top_recommender.recommend(user_id_array=user_id_array, cutoff=cutoff * 2,
                                                              remove_seen_flag=False))
        rec_true = np.array(top_recommender.recommend(user_id_array=user_id_array, cutoff=cutoff,
                                                      remove_seen_flag=True))

        # Get rid of common recommendations
        mask = [np.isin(double_rec_false[i], rec_true[i], invert=True) for i in range(double_rec_false.shape[0])]
        recommendations = np.zeros(shape=rec_true.shape)
        for i in range(recommendations.shape[0]):
            recommendations[i] = double_rec_false[i][mask[i]][0:cutoff]

    user_recommendations_items = recommendations.reshape((recommendations.size, 1)).squeeze()
    user_recommendations_user_id = np.repeat(user_id_array, repeats=cutoff)

    data_frame = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items},
                              dtype=np.int)

    return data_frame
