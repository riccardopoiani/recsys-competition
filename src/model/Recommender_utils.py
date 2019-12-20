
def apply_feature_weighting(matrix, feature_weighting="none"):
    import numpy as np
    from course_lib.Base.IR_feature_weighting import okapi_BM_25, TF_IDF
    from course_lib.Base.Recommender_utils import check_matrix

    FEATURE_WEIGHTING_VALUES = ["BM25", "TF-IDF", "none"]

    if feature_weighting not in FEATURE_WEIGHTING_VALUES:
        raise ValueError(
            "Value for 'feature_weighting' not recognized. Acceptable values are {}, provided was '{}'".format(
                FEATURE_WEIGHTING_VALUES, feature_weighting))

    if feature_weighting == "BM25":
        matrix = matrix.astype(np.float32)
        matrix = okapi_BM_25(matrix)
        matrix = check_matrix(matrix, 'csr')
    elif feature_weighting == "TF-IDF":
        matrix = matrix.astype(np.float32)
        matrix = TF_IDF(matrix)
        matrix = check_matrix(matrix, 'csr')
    return matrix