from course_lib.Base.BaseRecommender import BaseRecommender
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
import xgboost


def add_item_popularity():
    raise NotImplemented()


def add_user_len_information(data_frame: pd.DataFrame, URM_train):
    raise NotImplemented()


def remap_data_frame(df: pd.DataFrame, mapper):
    df = df.copy()
    raise NotImplemented()


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
    data_frame = data_frame.copy()
    df_region: pd.DataFrame = pd.read_csv(path + "data_UCM_age.csv")
    df_age: pd.DataFrame = pd.read_csv(path + "data_UCM_region.csv")

    # Re-map UCM data frame in order to have the correct user information
    if use_region:
        df_region = remap_data_frame(df=df_region, mapper=user_mapper)
        df_region = df_region[['row', 'col']]
        df_region = df_region.rename(columns={"col": "region"})
        data_frame = pd.merge(data_frame, df_region, right_on="row", left_on="user_id")
        data_frame = data_frame.drop(columns=["row"], inplace=True)
    if use_age:
        df_age = remap_data_frame(df=df_age, mapper=user_mapper)
        df_age = df_age[['row', 'col']]
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


def get_boosting_base_dataframe(URM_train: csr_matrix, top_recommender: BaseRecommender, cutoff: int):
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
    recommendations = np.array(top_recommender.recommend(user_id_array=np.arange(URM_train.shape[0]), cutoff=cutoff))
    user_recommendations_items = recommendations.reshape((recommendations.size, 1)).squeeze()

    # Setting users
    user_recommendations_user_id = np.zeros(user_recommendations_items.size)
    for user in range(0, URM_train.shape[0]):
        user_recommendations_user_id[(user * cutoff):(user * cutoff) + cutoff] = user

    data_frame = pd.DataFrame({"user_id": user_recommendations_user_id, "item_id": user_recommendations_items})

    return data_frame


class Boosting(BaseRecommender):

    def __init__(self, URM_train, dataframe):
        super().__init__(URM_train)

    def _compute_item_score(self, user_id_array, items_to_compute=None):
        raise NotImplemented()

    def save_model(self, folder_path, file_name=None):
        raise NotImplemented()
