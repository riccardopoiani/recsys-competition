from course_lib.Base.BaseRecommender import BaseRecommender
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from src.utils.general_utility_functions import get_total_number_of_users
from sklearn.model_selection import train_test_split
from src.utils.general_utility_functions import get_split_seed

import xgboost as xgb


def add_label(data_frame: pd.DataFrame, URM_train: csr_matrix):
    """
    Create a dataframe with a single column with the correct predictions

    :param data_frame: data frame containing information for boosting
    :param URM_train: URM train matrix
    :return: numpy array containing y information
    """
    user_ids = data_frame['user_id'].values
    item_ids = data_frame['item_id'].values

    y = np.zeros(user_ids.size)

    for i in range(0, user_ids.size):
        if i % 5000 == 0:
            print("{} done over {}".format(i, user_ids.size))
        true_ratings = URM_train[user_ids[i]].indices
        if item_ids[i] in true_ratings:
            y[i] = 1
        else:
            y[i] = 0

    non_zero_count = np.count_nonzero(y)
    print("There are {} non-zero ratings in {}".format(non_zero_count, y.size))

    return y


def add_user_len_information(data_frame: pd.DataFrame, URM_train: csr_matrix):
    """
    Add information concerning the user profile length to the row of the dataframe

    :param data_frame: data frame that is being pre-processed from boosting
    :param URM_train: URM train from which to take profile length information
    :return: data frame with new content inserted
    """
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


def add_UCM_information(data_frame: pd.DataFrame, user_mapper, path="../../data/", use_region=True, use_age=True):
    """
    Add UCM information to the data frame for XGboost

    :param data_frame: data frame containing information being pre-processed for boosting
    :param user_mapper: mapper original users to train users
    :param path: where to read UCM csv files
    :param use_region: True is region information should be used, false otherwise
    :param use_age: True if age information should be used, false otherwise
    :return: pd.DataFrame containing the original data frame+ UCM information
    """
    total_users = get_total_number_of_users()  # Total number of users (-1 since indexing from 0)

    data_frame = data_frame.copy()
    df_region: pd.DataFrame = pd.read_csv(path + "data_UCM_age.csv")
    df_age: pd.DataFrame = pd.read_csv(path + "data_UCM_region.csv")

    # Re-map UCM data frame in order to have the correct user information
    if use_region:
        df_region = df_region[['row', 'col']]

        dfDummies = pd.get_dummies(df_region['col'], prefix='region')
        dfDummies = dfDummies.join(df_region['row'])
        dfDummies = dfDummies.groupby(['row'], as_index=False).sum()

        # Fill missing values
        user_present = dfDummies['row'].values
        total_users = np.arange(total_users)
        mask = np.in1d(total_users, user_present, invert=True)
        missing_users = total_users[mask]
        num_col = dfDummies.columns.size
        missing_val = np.zeros(shape=(num_col, missing_users.size))
        missing_val[0] = missing_users
        missing_df = pd.DataFrame(data=np.transpose(missing_val), dtype=np.int32, columns=dfDummies.columns)
        df_region = dfDummies.append(missing_df)

        df_region = remap_data_frame(df=df_region, mapper=user_mapper)
        data_frame = pd.merge(data_frame, df_region, right_on="row", left_on="user_id")
        data_frame = data_frame.drop(columns=["row"], inplace=True)
    if use_age:
        df_age = df_age[['row', 'col']]

        # Handle missing values: fill with mode + 1
        users_present = df_age['row'].values
        total_users = np.arange(total_users)
        mask = np.in1d(total_users, users_present, invert=True)
        missing_users = total_users[mask].astype(np.int32)
        missing_val_filled = np.ones(missing_users.size) * (int(df_age['col'].mode()) + 1)
        missing = np.array([missing_users, missing_val_filled], dtype=np.int32)
        missing_df = pd.DataFrame(data=np.transpose(missing), columns=["row", "col"])
        df_age = df_age.append(missing_df)
        df_age = df_age.reset_index()
        df_age = df_age[['row', 'col']]

        df_age = remap_data_frame(df=df_age, mapper=user_mapper)
        df_age = df_age.rename(columns={"col": "age"})
        data_frame = pd.merge(data_frame, df_age, right_on="row", left_on="user_id")
        data_frame = data_frame.drop(columns=["row"], inplace=True)

    return data_frame


def add_ICM_information(data_frame: pd.DataFrame, path="../../data/", use_price=True, use_asset=True,
                        use_subclass=True):
    """
    Add information form the ICM files to the data frame

    :param data_frame: data frame that is being pre-processed for boosting
    :param path: path to the folder containing the csv files
    :param use_price: True if you wish to append price information, false otherwise
    :param use_asset: True if you wish to append asset information, false otherwise
    :param use_subclass: True if you wish to append subclass information, false otherwise
    :return: pd.DataFrame containing the information
    """
    data_frame = data_frame.copy()
    df_price: pd.DataFrame = pd.read_csv(path + "data_ICM_price.csv")
    df_asset: pd.DataFrame = pd.read_csv(path + "data_ICM_asset.csv")
    df_subclass: pd.DataFrame = pd.read_csv(path + "data_ICM_sub_class.csv")

    if use_price:
        df_price = df_price[['row', 'data']]
        df_price = df_price.rename(columns={"data": "price"})
        data_frame = pd.merge(data_frame, df_price, right_on="row", left_on="item_id")
        data_frame = data_frame.drop(columns=['row'], inplace=False)
    if use_asset:
        df_asset = df_asset[['row', 'data']]
        df_asset = df_asset.rename(columns={"data": "asset"})
        data_frame = pd.merge(data_frame, df_asset, right_on="row", left_on="item_id")
        data_frame = data_frame.drop(columns=["row"], inplace=False)
    if use_subclass:
        df_subclass = df_subclass[['row', 'col']]
        df_subclass = df_subclass.rename(columns={"col": "subclass"})
        data_frame = pd.merge(data_frame, df_subclass, right_on="row", left_on="item_id")
        data_frame = data_frame.drop(columns=["row"], inplace=False)

    return data_frame


def add_recommender_predictions(data_frame: pd.DataFrame, recommender: BaseRecommender, cutoff,
                                column_name):
    """
    Add recommender predictions to the data frame

    Note: this method assumes that in the current data frame, information has been added in the following way:
    "cutoff" predictions for each user

    :param data_frame: data frame on which info will be added
    :param recommender: recommender object from which scores will be predicted
    :param cutoff: cutoff with which the dataframe has been build
    :param column_name: name of the column that will be added
    :return:
    """
    df = data_frame.copy()

    cutoff = 20
    items = df['item_id'].values.astype(int)
    users = df['user_id'].values.astype(int)
    score_list = []

    for user_id in np.unique(users):
        items_for_user_id = items[(user_id * cutoff):(user_id * cutoff) + cutoff]
        scores = recommender._compute_item_score([user_id]).squeeze()[items_for_user_id]
        score_list.extend(scores)

    df[column_name] = pd.Series(score_list, index=df.index)

    return df


def get_boosting_base_dataframe(URM_train: csr_matrix, top_recommender: BaseRecommender, cutoff: int,
                                exclude_seen=False):
    """
    Get boosting data-frame preprocessed

    :param URM_train:
    :param ICM_train:
    :param UCM_train:
    :param top_recommender: top recommender used for building the dataframe
    :param cutoff: if you are interested in MAP@10, choose a large number, for instance, 20
    :return:
    """
    # Setting items
    recommendations = np.array(top_recommender.recommend(user_id_array=np.arange(URM_train.shape[0]), cutoff=cutoff,
                                                         remove_seen_flag=exclude_seen))
    user_recommendations_items = recommendations.reshape((recommendations.size, 1)).squeeze()

    # Setting users
    user_recommendations_user_id = np.zeros(user_recommendations_items.size)
    for user in range(0, URM_train.shape[0]):
        if user % 5000 == 0:
            print("{} done over {}".format(user, URM_train.shape[0]))
        user_recommendations_user_id[(user * cutoff):(user * cutoff) + cutoff] = user

    data_frame = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

    return data_frame


class Boosting(BaseRecommender):

    def __init__(self, URM_train, x, y, test_size=0.2):
        super().__init__(URM_train)
        X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=get_split_seed())
        self.dtrain = xgb.DMatrix(data=X_train, label=y_train)
        self.dtest = xgb.DMatrix(data=X_test, label=y_test)
        self.evallist = [(self.dtest, 'eval'), (self.dtrain, 'train')]

    def train(self, num_round, param, early_stopping_round):
        bst = xgb.train(params=param, dtrain=self.dtrain, num_boost_round=num_round, evals=self.evallist,
                        early_stopping_rounds=early_stopping_round)
        return bst

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        raise NotImplemented()

    def save_model(self, folder_path, file_name=None):
        raise NotImplemented()
