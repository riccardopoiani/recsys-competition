from src.model.Interface import IContentModel, IBestModel, ICollaborativeModel
import scipy.sparse as sps


class NewUserCF(ICollaborativeModel):
    """
    - MAP TUNING 5 FOLD: 0.026830�0.0016
    - MAP 10 FOLD: MAP: 0.0272575�0.0022
    (vs new_best_models on lt_23 on 10 FOLDS: MAP: 0.0264650�0.0020)
    """
    from src.model.KNN.NewUserKNNCFRecommender import NewUserKNNCFRecommender

    best_parameters = {'topK': 1295, 'shrink': 43, 'normalize': True, 'similarity': 'asymmetric',
                       'asymmetric_alpha': 0.007806844711984786, 'feature_weighting': 'BM25'}
    recommender_class = NewUserKNNCFRecommender
    recommender_name = "NewUserCF"


class RP3Beta_side_info(IBestModel):
    """
    RP3 beta with side info by using TF_IDF([URM_train, ICM_all.T])

    - MAP 5 FOLD: 0.035434�0.0013
    - MAP 10 FOLD: 0.0328794�0.0017
    """

    best_parameters = {'topK': 28, 'alpha': 0.008124745090408949,
                       'beta': 0.0051792301071096345, 'normalize_similarity': True}

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        from course_lib.Base.IR_feature_weighting import TF_IDF
        URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        from course_lib.GraphBased.RP3betaRecommender import RP3betaRecommender
        model = RP3betaRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class ItemCBF_CF(IContentModel):
    """
    Item CBF_CF tuned with URM_train and ICM_all  (containing sub_class, price, asset, sub_class_count, item_pop with
    discretization bins 200, 200, 50, 50)

     - MAP TUNING X-VAL (lt 23): 0.0365+-0.0019

     Further comments:
     - Twersky similarity is also very good, since it reaches 0.03654
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
    best_parameters = param = {'topK': 5, 'shrink': 1999, 'normalize': False, 'interactions_feature_weighting': 'BM25',
                               'similarity': 'tversky', 'tversky_alpha': 0.008230602341193551,
                               'tversky_beta': 1.574630363808344}

    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"


class Item_CBF_all(IContentModel):
    """
    X VALID 10 FOLDS: MAP: 0.0106720�0.0014 (LT 23)
    Basically the same of new_best_models
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 5, 'shrink': 19, 'normalize': False, 'similarity': 'asymmetric',
                       'asymmetric_alpha': 1.8793950443550476, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_all"


class Item_CBF_all_Bad(IContentModel):
    """
    MAP X VALID 10 FOLDS: MAP: 0.0076929�0.0008
    SUPER OVER-FIT FROM KEEP K OUT 3
    """
    from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender

    best_parameters = {'topK': 1, 'shrink': 0, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'none'}
    recommender_class = ItemKNNCBFRecommender
    recommender_name = "ItemCBF_all"


class NewItem_CBF(IContentModel):
    """
    MAP X VAL 10 FOLDS lt 23 : MAP: 0.0092201�0.0011

    VS new_best_models ItemCBF_all: MAP: 0.0107254�0.0010
    """
    from src.model.KNN.NewItemKNNCBFRecommender import NewItemKNNCBFRecommender

    recommender_class = NewItemKNNCBFRecommender
    recommender_name = "NewItemCBF"
    best_parameters = {'topK': 1, 'shrink': 2000, 'normalize': False, 'interactions_feature_weighting': 'none'}


class ItemCBF_all_FW(IBestModel):
    """
    MAP lt 23: 0.0088 (OF CFW)

    MAP 10 FOLD LT 23 OF THIS MODEL:

    MAP 10 FOLD LT 23 OF NEW BEST MODELS IS: MAP: 0.0114828�0.0011
    """

    CFW_parameters = {'topK': 800, 'add_zeros_quota': 0.008099525222030425, 'normalize_similarity': False}

    best_parameters = {'topK': 5, 'shrink': 1500, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.0, 'feature_weighting': 'TF-IDF'}

    @classmethod
    def get_model(cls, URM_train, ICM_train):
        from course_lib.KNN.ItemKNNCBFRecommender import ItemKNNCBFRecommender
        from course_lib.FeatureWeighting.CFW_D_Similarity_Linalg import CFW_D_Similarity_Linalg
        from src.model import best_models

        item_cf = best_models.ItemCF.get_model(URM_train=URM_train)
        cfw = CFW_D_Similarity_Linalg(URM_train=URM_train, ICM=ICM_train, S_matrix_target=item_cf.W_sparse)
        cfw.fit(**cls.CFW_parameters)

        model = ItemKNNCBFRecommender(URM_train=URM_train, ICM_train=ICM_train)
        model.fit(row_weights=cfw.D_best, **cls.get_best_parameters())
        return model
