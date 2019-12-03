from abc import ABC

import scipy.sparse as sps
import numpy as np


class FeatureWeightingStrategy(ABC):
    def weight_matrix(self, dataMatrix: sps.csr_matrix, feature_data):
        raise NotImplementedError("Method not implemented")


class LinearFeatureWeighting(FeatureWeightingStrategy):
    def weight_matrix(self, dataMatrix: sps.csr_matrix, feature_data):
        dataMatrix.data = dataMatrix.data * feature_data
        return dataMatrix


class LogFeatureWeighting(FeatureWeightingStrategy):
    def weight_matrix(self, dataMatrix: sps.csr_matrix, feature_data):
        feature_data[feature_data > 1] = np.log(feature_data[feature_data > 1])
        dataMatrix.data = dataMatrix.data * feature_data
        return dataMatrix


class InverseFeatureWeighting(FeatureWeightingStrategy):
    def weight_matrix(self, dataMatrix: sps.csr_matrix, feature_data):
        dataMatrix.data = dataMatrix.data * (1/feature_data)
        return dataMatrix


class InverseLog1pFeatureWeighting(FeatureWeightingStrategy):
    def weight_matrix(self, dataMatrix: sps.csr_matrix, feature_data):
        dataMatrix.data = dataMatrix.data * (1/np.log1p(feature_data))
        return dataMatrix


STRATEGY_MAPPER = {"linear": LinearFeatureWeighting(), "log": LogFeatureWeighting(),
                   "inverse": InverseFeatureWeighting(), "inverse_log1p": InverseLog1pFeatureWeighting()}

PROPORTIONAL_STRATEGY = ["linear", "log"]


def _weight_matrix(dataMatrix: sps.csr_matrix, weights: np.ndarray, strategy="linear"):
    """
    Assuming that rows are the object to weights, it returns a weighted matrix based on the weights and alpha

    :param dataMatrix: dataMatrix with rows being the object to weight
    :param weights: an array with the same length as the number of nnz inside the dataMatrix
    :param strategy: strategy to use in order to put weights
    :return: new data matrix with weighted data
    """
    if len(weights) != len(dataMatrix.data):
        raise ValueError("Demographic list does not contain all users in dataMatrix")

    matrix = dataMatrix.tocsr(copy=True)
    strategy_funct = STRATEGY_MAPPER[strategy]
    matrix = strategy_funct.weight_matrix(matrix, feature_data=weights)
    return sps.csr_matrix(matrix)


def weight_matrix_by_user_feature_counts(dataMatrix: sps.csr_matrix, UCM: sps.csr_matrix, strategy="linear"):
    """
    Assumes that rows of dataMatrix are users and it returns the weighted dataMatrix based on user feature counts of
    UCM

    :param dataMatrix:
    :param UCM:
    :param strategy: strategy to use in order to put weights
    :return:
    """
    if UCM.shape[0] != dataMatrix.shape[0]:
        raise ValueError("UCM does not contain all users in dataMatrix")

    UCM_popularity = (UCM > 0).sum(axis=0)
    UCM_popularity = np.array(UCM_popularity).squeeze()
    user_list = UCM.tocoo().row
    feature_list = UCM.tocoo().col

    user_feature_list = np.full(shape=dataMatrix.shape[0], fill_value=1)
    user_feature_list[user_list] = UCM_popularity[feature_list]

    users = dataMatrix.tocoo().row
    feature_list_for_user = np.array(user_feature_list[users], dtype=np.float32)

    return _weight_matrix(dataMatrix, feature_list_for_user, strategy)


def weight_matrix_by_item_feature_value(dataMatrix: sps.csr_matrix, ICM: sps.csr_matrix, strategy="linear"):
    """
    Assumes that rows of dataMatrix are items and it returns the weighted dataMatrix based on item feature value
    of ICM

    :param dataMatrix:
    :param ICM: ICM with only one columns
    :param strategy:
    :return:
    """
    if ICM.shape[0] != dataMatrix.shape[0]:
        raise ValueError("ICM does not contain all items in dataMatrix")

    item_list = dataMatrix.tocoo().row
    item_feature_weights = np.array(ICM[item_list].todense()).squeeze()
    mean_value = item_feature_weights.mean()
    item_feature_weights[item_feature_weights == 0] = mean_value

    return _weight_matrix(dataMatrix, item_feature_weights, strategy)

def weight_matrix_by_user_profile(dataMatrix: sps.csr_matrix, URM, strategy="linear"):
    """

    :param dataMatrix:
    :param URM:
    :param strategy: strategy to use in order to put weights
    :return:
    """
    if URM.shape[0] != dataMatrix.shape[0]:
        raise ValueError("URM does not contain all users in dataMatrix")

    user_activity = (URM > 0).sum(axis=1)
    user_activity = np.array(user_activity).squeeze()

    users = dataMatrix.tocoo().row
    user_profile_for_user = np.array(user_activity[users], dtype=np.float32)

    return _weight_matrix(dataMatrix, user_profile_for_user, strategy)


def weight_matrix_by_item_popularity(dataMatrix: sps.csr_matrix, R_iu: sps.csr_matrix, strategy="linear"):
    """
    Assumes that dataMatrix has items as row

    :param dataMatrix: csr matrix with items as rows
    :param R_iu: Rating item x user matrix
    :param strategy: strategy to use in order to put weights
    :return:
    """
    if R_iu.shape[0] != dataMatrix.shape[0]:
        raise ValueError("R_iu does not contain all items in dataMatrix")

    item_popularity = (R_iu > 0).sum(axis=1)
    item_popularity = np.array(item_popularity).squeeze()

    items = dataMatrix.tocoo().row
    item_popularity_for_item = np.array(item_popularity[items], dtype=np.float32)

    return _weight_matrix(dataMatrix, item_popularity_for_item, strategy)


