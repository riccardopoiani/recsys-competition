import numpy as np
from scipy.sparse import csr_matrix


def get_age_rating_matrix(URM, age_demographic, implicit=False):
    ratings_per_age = np.zeros(shape=(len(age_demographic), URM.shape[1]))
    for i, users_age in enumerate(age_demographic):
        URM_age = URM[users_age].copy()
        ratings_age = np.squeeze(np.asarray(URM_age.sum(axis=0)))
        if implicit:
            ratings_age[ratings_age > 1] = 1
        ratings_per_age[i] = ratings_age

    age_rating_matrix = csr_matrix(ratings_per_age, dtype=np.float64)

    return age_rating_matrix


def get_subclass_rating_matrix(URM, subclass_content_dict, implicit=False):
    subclasses = list(subclass_content_dict.keys())
    ratings_per_subclass = np.zeros(shape=(len(subclasses), URM.shape[0]))

    URM_csc = URM.tocsc()

    for i, sub in enumerate(subclasses):
        items = subclass_content_dict[sub]
        URM_subclass = URM_csc[:, items]
        ratings_subclass = np.squeeze(np.asarray(URM_subclass.sum(axis=1)))
        if implicit:
            ratings_subclass[ratings_subclass > 1] = 1
        ratings_per_subclass[i] = ratings_subclass

    ratings_per_subclass = np.transpose(ratings_per_subclass)

    subclass_rating_matrix = csr_matrix(ratings_per_subclass, dtype=np.float64)

    return subclass_rating_matrix
