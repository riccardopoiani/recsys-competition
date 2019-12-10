import numpy as np
import scipy.sparse as sps

from course_lib.Data_manager.IncrementalSparseMatrix import IncrementalSparseMatrix
from src.data_management.RecSys2019Reader_utils import build_UCM_all
from src.feature.feature_processing import transform_numerical_to_label


def apply_feature_engineering_UCM(UCM_dict, URM, ICM_dict: dict, ICM_names_to_UCM: list):
    if ~np.all(np.in1d(ICM_names_to_UCM, list(ICM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for ICM_name in ICM_names_to_UCM:
        ICM_object = ICM_dict[ICM_name]
        ICM_suffix_name = ICM_name.replace("ICM", "")

        new_UCM = URM.dot(ICM_object).tocoo()
        new_UCM_name = "UCM{}".format(ICM_suffix_name)

        UCM_dict[new_UCM_name] = new_UCM
    return UCM_dict


def apply_transformation_UCM(UCM_dict, UCM_name_to_transform_mapper: dict):
    if ~np.all(np.in1d(list(UCM_name_to_transform_mapper.keys()), list(UCM_dict.keys()))):
        raise KeyError("Mapper contains wrong UCM names")

    for UCM_name, transformer in UCM_name_to_transform_mapper.items():
        UCM_object: sps.csr_matrix = UCM_dict[UCM_name]
        UCM_object.data = transformer(UCM_object.data)
        UCM_dict[UCM_name] = UCM_object
    return UCM_dict


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


def build_UCM_all_from_dict(UCM_dict: dict):
    mapper_dict = {}
    for UCM_name in UCM_dict.keys():
        mapper_dict[UCM_name] = {}
    UCM_all, _ = build_UCM_all(UCM_dict, mapper_dict)
    return UCM_all
