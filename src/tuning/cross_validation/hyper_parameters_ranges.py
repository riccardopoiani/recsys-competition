from skopt.space import Integer, Real, Categorical

from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.model.FactorizationMachine.FieldAwareFMRecommender import FieldAwareFMRecommender
from src.model.MatrixFactorization.FunkSVDRecommender import FunkSVDRecommender
from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
from src.model.MatrixFactorization.LightFMRecommender import LightFMRecommender
from src.model.MatrixFactorization.LogisticMFRecommender import LogisticMFRecommender
from src.model.MatrixFactorization.MF_BPR_Recommender import MF_BPR_Recommender

# ------------------------- HYPER PARAMETERS RANGE ------------------------- #
HYPER_PARAMETERS_RANGE = {
    ImplicitALSRecommender.RECOMMENDER_NAME: {
        "num_factors": Integer(300, 550),
        "regularization": Real(low=1e-2, high=200, prior='log-uniform'),
        "epochs": Categorical([50]),
        "confidence_scaling": Categorical(["linear"]),
        "alpha": Real(low=1e-2, high=1e2, prior='log-uniform')
    },

    MF_BPR_Recommender.RECOMMENDER_NAME: {
        "num_factors": Categorical([600]),
        "regularization": Real(low=1e-4, high=1e-1, prior='log-uniform'),
        "learning_rate": Real(low=1e-2, high=1e-1, prior='log-uniform'),
        "epochs": Categorical([300])
    },

    FunkSVDRecommender.RECOMMENDER_NAME: {
        "num_factors": Integer(50, 400),
        "regularization": Real(low=1e-5, high=1e-0, prior='log-uniform'),
        "learning_rate": Real(low=1e-2, high=1e-1, prior='log-uniform'),
        "epochs": Categorical([500])
    },

    LogisticMFRecommender.RECOMMENDER_NAME: {
        "num_factors": Integer(20, 400),
        "regularization": Real(low=1e-5, high=1e1, prior='log-uniform'),
        "learning_rate": Real(low=1e-2, high=1e-1, prior='log-uniform'),
        "epochs": Categorical([300])
    },

    LightFMRecommender.RECOMMENDER_NAME: {
        'no_components': Categorical([100]),
        'epochs': Categorical([100])
    },

    FieldAwareFMRecommender.RECOMMENDER_NAME: {
        'epochs': Categorical([200]),
        'latent_factors': Integer(low=20, high=500),
        'regularization': Real(low=10e-7, high=10e-1, prior="log-uniform"),
        'learning_rate': Real(low=10e-3, high=10e-1, prior="log-uniform")
    },

    ItemKNNCFRecommender.RECOMMENDER_NAME: {
        "topK": Integer(5, 1000),
        "shrink": Integer(0, 2000),
        "normalize": Categorical([True, False])
    },

    P3alphaRecommender.RECOMMENDER_NAME: {
        "topK": Integer(1, 1000),
        "alpha": Real(low=0, high=2, prior='uniform'),
        "normalize_similarity": Categorical([True, False])
    }

}


def get_hyper_parameters_dictionary(recommender_class):
    hyperparameters_range_dictionary = {}
    return HYPER_PARAMETERS_RANGE[recommender_class.RECOMMENDER_NAME]


def add_knn_similarity_type_hyper_parameters(hyper_parameters_dictionary: dict, similarity_type: str,
                                             allow_weighting=True):
    hyper_parameters_dictionary["similarity"] = Categorical([similarity_type])
    is_set_similarity = similarity_type in ["tversky", "dice", "jaccard", "tanimoto"]

    if similarity_type == "asymmetric":
        hyper_parameters_dictionary["asymmetric_alpha"] = Real(low=0, high=2, prior='uniform')
        hyper_parameters_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "tversky":
        hyper_parameters_dictionary["tversky_alpha"] = Real(low=0, high=2, prior='uniform')
        hyper_parameters_dictionary["tversky_beta"] = Real(low=0, high=2, prior='uniform')
        hyper_parameters_dictionary["normalize"] = Categorical([True])

    elif similarity_type == "euclidean":
        hyper_parameters_dictionary["normalize"] = Categorical([True, False])
        hyper_parameters_dictionary["normalize_avg_row"] = Categorical([True, False])
        hyper_parameters_dictionary["similarity_from_distance_mode"] = Categorical(["lin", "log", "exp"])

    if not is_set_similarity:
        if allow_weighting:
            hyper_parameters_dictionary["feature_weighting"] = Categorical(["none", "BM25", "TF-IDF"])
    return hyper_parameters_dictionary
