import numpy as np
import scipy.sparse as sps

from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from src.data_management.RecSys2019Reader_utils import build_UCM_all


# ----------- FEATURE ENGINEERING -----------


def apply_feature_engineering_ICM(ICM_dict: dict, URM, UCM_dict: dict, ICM_names_to_count: list,
                                  UCM_names_to_list: list):
    if ~np.all(np.in1d(list(ICM_names_to_count), list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong ICM names")

    if ~np.all(np.in1d(UCM_names_to_list, list(UCM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for ICM_name in ICM_names_to_count:
        ICM_object: sps.csr_matrix = ICM_dict[ICM_name]
        column = ICM_object.tocoo().col
        uniques, counts = np.unique(column, return_counts=True)

        new_ICM_name = "{}_count".format(ICM_name)
        new_row = np.array(ICM_object.tocoo().row, dtype=int)
        new_col = np.array([0] * len(new_row), dtype=int)
        new_data = np.array(counts[column], dtype=np.float32)

        ICM_builder = IncrementalSparseMatrix()
        ICM_builder.add_data_lists(new_row, new_col, new_data)
        ICM_dict[new_ICM_name] = ICM_builder.get_SparseMatrix()

    for UCM_name in UCM_names_to_list:
        UCM_object = UCM_dict[UCM_name]
        UCM_suffix_name = UCM_name.replace("UCM", "")

        new_ICM = URM.T.dot(UCM_object)
        new_ICM_name = "ICM{}".format(UCM_suffix_name)

        ICM_dict[new_ICM_name] = new_ICM.tocsr()
    return ICM_dict


def apply_feature_engineering_UCM(UCM_dict: dict, URM, ICM_dict: dict, ICM_names_to_UCM: list):
    if ~np.all(np.in1d(ICM_names_to_UCM, list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for ICM_name in ICM_names_to_UCM:
        ICM_object = ICM_dict[ICM_name]
        ICM_suffix_name = ICM_name.replace("ICM", "")

        new_UCM = URM.dot(ICM_object)
        new_UCM_name = "UCM{}".format(ICM_suffix_name)

        UCM_dict[new_UCM_name] = new_UCM.tocsr()
    return UCM_dict


def apply_feature_entropy_UCM(UCM_dict: dict, UCM_names_to_entropy: list):
    if ~np.all(np.in1d(UCM_names_to_entropy, list(UCM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for UCM_name in UCM_names_to_entropy:
        UCM_object: sps.csr_matrix = UCM_dict[UCM_name]

        total_interactions_each_row = np.array(UCM_object.sum(axis=1)).squeeze()
        interactions_each_row = UCM_object.indptr[1:] - UCM_object.indptr[:-1]
        total_interactions = np.repeat(total_interactions_each_row, interactions_each_row)
        UCM_object.data = UCM_object.data / total_interactions

        log_UCM_object = UCM_object.copy()
        log_UCM_object.data = np.log2(log_UCM_object.data)

        entropy_matrix = UCM_object.multiply(log_UCM_object)
        entropy = np.array(entropy_matrix.sum(axis=1)).squeeze()

        new_UCM_name = "{}_entropy".format(UCM_name)
        new_row = np.arange(UCM_object.shape[0])
        new_col = np.array([0] * len(new_row), dtype=int)
        new_data = entropy

        UCM_builder = IncrementalSparseMatrix()
        UCM_builder.add_data_lists(new_row, new_col, new_data)
        UCM_dict[new_UCM_name] = UCM_builder.get_SparseMatrix()

    return UCM_dict


# ----------- FEATURE IMPUTATION -----------

def apply_imputation_ICM(ICM_dict: dict, ICM_name_to_agg_mapper: dict):
    if ~np.all(np.in1d(list(ICM_name_to_agg_mapper.keys()), list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong ICM names")

    for ICM_name, aggregator in ICM_name_to_agg_mapper.items():
        ICM_object: sps.csr_matrix = ICM_dict[ICM_name]
        x = np.array(ICM_object.data)
        missing_x_mask = np.ediff1d(ICM_object.tocsr().indptr) == 0
        missing_x = np.arange(ICM_object.shape[0])[missing_x_mask]
        ICM_object = ICM_object.tocoo()
        ICM_object.row = np.concatenate([ICM_object.row, missing_x])
        sort_indices = np.argsort(ICM_object.row)
        ICM_object.row = ICM_object.row[sort_indices]
        ICM_object.col = np.concatenate([ICM_object.col, [ICM_object.col[0]] * len(missing_x)])
        ICM_object.data = np.concatenate([ICM_object.data, [aggregator(x)] * len(missing_x)])
        ICM_object.data = ICM_object.data[sort_indices]
        ICM_dict[ICM_name] = ICM_object.tocsr()
    return ICM_dict


def apply_imputation_UCM(UCM_dict: dict, UCM_name_to_agg_mapper: dict):
    if ~np.all(np.in1d(list(UCM_name_to_agg_mapper.keys()), list(UCM_dict.keys()))):
        raise KeyError("Mapper contains wrong ICM names")

    for UCM_name, aggregator in UCM_name_to_agg_mapper.items():
        UCM_object: sps.csr_matrix = UCM_dict[UCM_name]
        missing_x_mask = np.ediff1d(UCM_object.tocsr().indptr) == 0
        missing_x = np.arange(UCM_object.shape[0])[missing_x_mask]

        UCM_object = UCM_object.tocoo()
        UCM_object.row = np.concatenate([UCM_object.row, missing_x])
        UCM_object.col = np.concatenate([UCM_object.col, [aggregator(UCM_object.col)] * len(missing_x)])
        UCM_object.data = np.concatenate([UCM_object.data, np.ones(len(missing_x))])
        UCM_dict[UCM_name] = UCM_object.tocsr()
    return UCM_dict


# ----------- FEATURE FILTERING -----------

def apply_filtering_ICM(ICM_dict: dict, ICM_name_to_filter_mapper: dict):
    if ~np.all(np.in1d(list(ICM_name_to_filter_mapper.keys()), list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong ICM names")

    for ICM_name, filter in ICM_name_to_filter_mapper.items():
        ICM_object: sps.csr_matrix = ICM_dict[ICM_name]
        mask = filter(ICM_object.data)
        ICM_object.data[~mask] = 0
        ICM_object.eliminate_zeros()
        ICM_dict[ICM_name] = ICM_object
    return ICM_dict


# ----------- FEATURE TRANSFORMATION -----------

def apply_transformation_ICM(ICM_dict, ICM_name_to_transform_mapper: dict):
    if ~np.all(np.in1d(list(ICM_name_to_transform_mapper.keys()), list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong ICM names")

    for ICM_name, transformer in ICM_name_to_transform_mapper.items():
        ICM_object: sps.csr_matrix = ICM_dict[ICM_name]
        ICM_object.data = np.array(transformer(ICM_object.data), dtype=np.float32)
        ICM_dict[ICM_name] = ICM_object
    return ICM_dict


def apply_transformation_UCM(UCM_dict, UCM_name_to_transform_mapper: dict):
    if ~np.all(np.in1d(list(UCM_name_to_transform_mapper.keys()), list(UCM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for UCM_name, transformer in UCM_name_to_transform_mapper.items():
        UCM_object: sps.csr_matrix = UCM_dict[UCM_name]
        UCM_object.data = np.array(transformer(UCM_object.data), dtype=np.float32)
        UCM_dict[UCM_name] = UCM_object
    return UCM_dict


# ----------- FEATURE DISCRETIZATION -----------

def apply_discretization_ICM(ICM_dict, ICM_name_to_bins_mapper: dict):
    if ~np.all(np.in1d(list(ICM_name_to_bins_mapper.keys()), list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for ICM_name, bins in ICM_name_to_bins_mapper.items():
        ICM_object: sps.csr_matrix = ICM_dict[ICM_name]
        if ICM_object.shape[1] != 1:
            raise KeyError("Given UCM name is not regarding a single feature, thus, it cannot be discretized")

        x = np.array(ICM_object.data)
        labelled_x = transform_numerical_to_label(x, bins)

        ICM_builder = IncrementalSparseMatrix(n_rows=ICM_object.shape[0])
        ICM_builder.add_data_lists(ICM_object.tocoo().row, labelled_x, np.ones(len(labelled_x), dtype=np.float32))

        ICM_dict[ICM_name] = ICM_builder.get_SparseMatrix()
    return ICM_dict


def apply_advanced_discretization_ICM(ICM_dict, ICM_name_to_bins_mapper: dict):
    if ~np.all(np.in1d(list(ICM_name_to_bins_mapper.keys()), list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for ICM_name, bins in ICM_name_to_bins_mapper.items():
        ICM_object: sps.csr_matrix = ICM_dict[ICM_name]
        if ICM_object.shape[1] != 1:
            raise KeyError("Given UCM name is not regarding a single feature, thus, it cannot be discretized")

        ICM_object = transform_numerical_to_discretized_sparse_matrix(ICM_object.tocoo().row,
                                                                             ICM_object.data, bins)
        ICM_dict[ICM_name] = ICM_object
    return ICM_dict


def apply_discretization_UCM(UCM_dict, UCM_name_to_bins_mapper: dict):
    if ~np.all(np.in1d(list(UCM_name_to_bins_mapper.keys()), list(UCM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for UCM_name, bins in UCM_name_to_bins_mapper.items():
        UCM_object: sps.csr_matrix = UCM_dict[UCM_name]
        if UCM_object.shape[1] != 1:
            raise KeyError("Given UCM name is not regarding a single feature, thus, it cannot be discretized")

        x = np.array(UCM_object.data)
        labelled_x = transform_numerical_to_label(x, bins)

        UCM_builder = IncrementalSparseMatrix(n_rows=UCM_object.shape[0])
        UCM_builder.add_data_lists(UCM_object.tocoo().row, labelled_x, np.ones(len(labelled_x), dtype=np.float32))

        UCM_dict[UCM_name] = UCM_builder.get_SparseMatrix()
    return UCM_dict


# ----------- BUILD ALL -----------

def build_ICM_all_from_dict(ICM_dict: dict):
    mapper_dict = {}
    for ICM_name in ICM_dict.keys():
        mapper_dict[ICM_name] = {}
    ICM_all, _ = build_UCM_all(ICM_dict, mapper_dict)
    return ICM_all


def build_UCM_all_from_dict(UCM_dict: dict):
    mapper_dict = {}
    for UCM_name in UCM_dict.keys():
        mapper_dict[UCM_name] = {}
    UCM_all, _ = build_UCM_all(UCM_dict, mapper_dict)
    return UCM_all


# ----------- UTILS -----------

def transform_numerical_to_label(x: np.ndarray, bins=20):
    """
    Given an array x, containing continuous numerical values, it digitizes the array into
    <<bins>> labels.

    :param x: array of numerical values
    :param bins: number of labels in the output
    :return: array of labelled values
    """
    eps = 10e-6
    norm_x = (x - x.min()) / (x.max() - x.min() + eps) * 100
    bins_list = [i * (norm_x.max() / bins) for i in range(bins)]
    labelled_x = np.digitize(norm_x, bins_list, right=True)
    return labelled_x


def transform_numerical_to_discretized_sparse_matrix(row: np.ndarray, data: np.ndarray, bins=20):
    """
    Transform a numerical data array into a discretized sparse matrix with a certain amount of bins

    :param data: array of numerical values
    :param bins: number of labels in the output
    :return: sparse matrix containing the discretized values
    """
    if row.size != data.size:
        raise ValueError("Row size has to be the same of data size")

    eps = 10e-6
    norm_x = (data - data.min()) / (data.max() - data.min() + eps) * 100
    step = (norm_x.max() / bins)
    bins_list = [i * step for i in range(bins)]

    sparse_matrix_builder = IncrementalSparseMatrix(n_rows=np.max(row)+1, n_cols=bins+1)
    for i, x in enumerate(norm_x):
        x_dist = np.random.normal(loc=x, scale=step, size=1000)
        label_x_dist = np.digitize(x_dist, bins_list, right=True)
        unique_label_x_dist, unique_counts = np.unique(label_x_dist, return_counts=True)
        unique_counts = unique_counts/np.max(unique_counts)  # Normalize unique counts
        size = unique_label_x_dist.size

        sparse_matrix_builder.add_data_lists([row[i]] * size, unique_label_x_dist, unique_counts)
    sparse_matrix = sparse_matrix_builder.get_SparseMatrix()
    return sparse_matrix
