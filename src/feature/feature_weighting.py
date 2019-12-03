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


def weight_matrix_by_users(dataMatrix: sps.csr_matrix, weights, strategy="linear"):
    """
    Assuming that rows are users, it returns a weighted matrix based on users and alpha

    :param strategy:
    :param weights:
    :param dataMatrix:
    :return: new data matrix with weighted data
    """
    if len(weights) != len(dataMatrix.data):
        raise ValueError("Demographic list does not contain all users in dataMatrix")

    matrix = dataMatrix.tocsr(copy=True)
    strategy_funct = STRATEGY_MAPPER[strategy]
    matrix = strategy_funct.weight_matrix(matrix, feature_data=weights)
    return sps.csr_matrix(matrix)


def weight_matrix_by_demographic_popularity(dataMatrix: sps.csr_matrix, UCM: sps.csr_matrix, strategy="linear"):
    """

    :param strategy:
    :param dataMatrix:
    :param UCM:
    :return:
    """
    if UCM.shape[0] != dataMatrix.shape[0]:
        raise ValueError("Demographic list does not contain all users in dataMatrix")

    UCM_popularity = (UCM > 0).sum(axis=0)
    UCM_popularity = np.array(UCM_popularity).squeeze()
    user_list = UCM.tocoo().row
    feature_list = UCM.tocoo().col

    user_feature_list = np.full(shape=dataMatrix.shape[0], fill_value=1)
    user_feature_list[user_list] = UCM_popularity[feature_list]

    users = dataMatrix.tocoo().row
    feature_list_for_user = np.array(user_feature_list[users], dtype=np.float32)

    return weight_matrix_by_users(dataMatrix, feature_list_for_user, strategy)

def weight_matrix_by_user_profile(dataMatrix: sps.csr_matrix, URM, strategy="linear"):
    """

    :param dataMatrix:
    :param URM:
    :param strategy:
    :return:
    """
    if URM.shape[0] != dataMatrix.shape[0]:
        raise ValueError("Demographic list does not contain all users in dataMatrix")

    user_activity = (URM > 0).sum(axis=1)
    user_activity = np.array(user_activity).squeeze()

    users = dataMatrix.tocoo().row
    user_profile_for_user = np.array(user_activity[users], dtype=np.float32)

    return weight_matrix_by_users(dataMatrix, user_profile_for_user, strategy)
