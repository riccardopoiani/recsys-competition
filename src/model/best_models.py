from src.model.Interface import IBestModel, ICollaborativeModel, IContentModel


# ---------------- CONTENT BASED FILTERING -----------------
class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM (containing sub_class and URM_train)
     - MAP on tuning: 0.0273
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 10, 'shrink': 1056, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.026039165670822324, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFRecommender


class ItemCBF_numerical(IContentModel):
    """
    Item CBF tuned with URM_train and ICM_numerical (containing price, asset and item popularity)
     - MAP on tuning: 0.002
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'feature_weighting': 'none', 'normalize': False, 'normalize_avg_row': True,
                       'shrink': 0, 'similarity': 'euclidean', 'similarity_from_distance_mode': 'exp',
                       'topK': 1000}
    recommender_class = ItemKNNCBFRecommender


class ItemCBF_categorical(IContentModel):
    """
    Item CBF tuned with URM_train and ICM_categorical (containing sub_class)
     - MAP on tuning: 0.0041
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 5, 'shrink': 1000, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 2.0, 'feature_weighting': 'BM25'}
    recommender_class = ItemKNNCBFRecommender


# ---------------- COLLABORATIVE FILTERING -----------------
class ItemCF(ICollaborativeModel):
    """
    Item CF tuned with URM_train
     - MAP on tuning: about 0.0262
    """
    from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender

    best_parameters = {'topK': 5, 'shrink': 1000, 'similarity': 'cosine', 'normalize': True,
                       'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCFRecommender


class UserCF(ICollaborativeModel):
    """
    User CF tuned with URM_train
     - MAP on tuning: about 0.019
    """
    from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender

    best_parameters = {'topK': 995, 'shrink': 9, 'similarity': 'cosine', 'normalize': True,
                       'feature_weighting': 'TF-IDF'}
    recommender_class = UserKNNCFRecommender


class P3Alpha(ICollaborativeModel):
    """
    P3Alpha recommender tuned with URM_train
     - MAP on tuning: 0.0247
    """
    from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender

    best_parameters = {'topK': 84, 'alpha': 0.6033770403001427, 'normalize_similarity': True}
    recommender_class = P3alphaRecommender


class RP3Beta(ICollaborativeModel):
    """
    RP3Beta recommender tuned with URM_train
     - MAP on tuning: 0.0241
    """
    from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender

    best_parameters = {'topK': 5, 'alpha': 0.37829128706576887, 'beta': 0.0, 'normalize_similarity': False}
    recommender_class = RP3betaRecommender


class SLIM_BPR(ICollaborativeModel):
    """
    SLIM_BPR recommender tuned with URM_train
     - There is still need to be tuned better
     - MAP on tuning: 0.0217
    """
    from course_lib.SLIM_BPR.Cython.SLIM_BPR_Cython import SLIM_BPR_Cython

    best_parameters = {'topK': 5, 'epochs': 1499, 'symmetric': False, 'sgd_mode': 'adagrad',
                       'lambda_i': 1e-05, 'lambda_j': 0.01, 'learning_rate': 0.0001}
    recommender_class = SLIM_BPR_Cython


class PureSVD(ICollaborativeModel):
    """
    PureSVD recommender tuned with URM_train
    - MAP on tuning: 0.0147
    """
    from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
    best_parameters = {'num_factors': 376}
    recommender_class = PureSVDRecommender


class IALS(ICollaborativeModel):
    """
    IALS recommender
     - MAP (only warm): 0.029
    """
    from src.model.MatrixFactorization.ImplicitALSRecommender import ImplicitALSRecommender
    best_parameters = {'epochs':100 ,'num_factors': 1000, 'regularization': 8}
    recommender_class = ImplicitALSRecommender


# ---------------- DEMOGRAPHIC FILTERING -----------------
class AdvancedTopPop(IBestModel):
    """
    Advanced Top Popular tuned with URM_train and UCM+user_profile_len
     - MAP on tuning (only cold users): 0.0087
    """
    best_parameters = {'clustering_method': 'kmodes', 'n_clusters': 45, 'init_method': 'random'}

    @classmethod
    def get_model(cls, URM_train, demographic_df, original_to_train_user_id_mapper):
        from src.model.FallbackRecommender.AdvancedTopPopular import AdvancedTopPopular
        model = AdvancedTopPopular(URM_train=URM_train, data=demographic_df,
                                   mapper_dict=original_to_train_user_id_mapper)
        model.fit(**cls.best_parameters)
        return model


class UserCBF(IBestModel):
    """
    User CBF tuned with URM_train and UCM (containing age, region and URM_train)
     - MAP on tuning (only cold users): 0.0109
    """
    best_parameters = {'topK': 3285, 'shrink': 1189, 'similarity': 'cosine',
                       'normalize': False, 'feature_weighting': 'BM25'}

    @classmethod
    def get_model(cls, URM_train, UCM_train):
        from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
        model = UserKNNCBFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**cls.get_best_parameters())
        return model


# ---------------- HYBRIDS -----------------

class HybridWeightedAvgSubmission1(IBestModel):
    """
    Hybrid Weighted Average with TopPop Fallback, done for the first submission of Kaggle competition
     - MAP on tuning (w/o TopPop Fallback): 0.0272
    """

    best_parameters = {'ITEM_CF': 0.969586046573504, 'USER_CF': 0.943330450168123,
                       'ITEM_CBF_NUM': 0.03250599212747674, 'ITEM_CBF_CAT': 0.018678076600871066,
                       'SLIM_BPR': 0.03591603993769955, 'P3ALPHA': 0.7474845972085382, 'RP3BETA': 0.1234024366177027}

    @classmethod
    def _get_all_models(cls, URM_train, ICM_numerical, ICM_categorical):
        all_models = {'ITEM_CF': ItemCF.get_model(URM_train=URM_train),
                      'USER_CF': UserCF.get_model(URM_train=URM_train),
                      'ITEM_CBF_NUM': ItemCBF_numerical.get_model(URM_train=URM_train, ICM_train=ICM_numerical),
                      'ITEM_CBF_CAT': ItemCBF_categorical.get_model(URM_train=URM_train, ICM_train=ICM_categorical),
                      'SLIM_BPR': SLIM_BPR.get_model(URM_train=URM_train),
                      'P3ALPHA': P3Alpha.get_model(URM_train=URM_train),
                      'RP3BETA': RP3Beta.get_model(URM_train=URM_train)}
        return all_models

    @classmethod
    def get_model(cls, URM_train, ICM_numerical, ICM_categorical):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender
        model = HybridWeightedAverageRecommender(URM_train=URM_train)
        all_models = cls._get_all_models(URM_train=URM_train, ICM_numerical=ICM_numerical,
                                         ICM_categorical=ICM_categorical)
        for model_name, model_object in all_models.items():
            model.add_fitted_model(model_name, model_object)
        model.fit(**cls.get_best_parameters())
        return model
