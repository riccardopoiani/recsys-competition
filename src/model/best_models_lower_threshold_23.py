from skopt.space import Categorical, Integer

from src.model.Interface import IContentModel, IBestModel, ICollaborativeModel
import scipy.sparse as sps


class WeightedAverageItemBasedWithoutRP3(IBestModel):
    """
    MAP 5: 0.0375
    MAP 10: 0.0370478�0.0019
    """
    best_parameters_true = {'FUSION': 0.14212574141484816, 'ItemDotCF': 0.9812875193125008,
                            'ItemCBF_CF': 0.9792817153741367}
    best_parameters = {'FUSION': 0.15, "ItemDotCF": 1, 'ItemCBF_CF': 1}

    @classmethod
    def get_model(cls, URM_train, ICM_all):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender

        fusion = FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)
        item_cbf_cf = ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)
        item_dot = ItemDotCF.get_model(URM_train=URM_train, load_model=False)

        hybrid = HybridWeightedAverageRecommender(URM_train, normalize=True)
        hybrid.add_fitted_model("ItemDotCF", item_dot)
        hybrid.add_fitted_model("FUSION", fusion)
        hybrid.add_fitted_model("ItemCBF_CF", item_cbf_cf)

        hybrid.fit(**cls.get_best_parameters())
        return hybrid


class WeightedAverageItemBasedWithRP3(IBestModel):
    """
    CV MAP 0.0374 TFIDF TRUE
    10 FOLD CV: 0.0373325�0.0018 TFIDF TRUE

    CV MAP TUNING TF IDF FALSE: 0.0377
    10 FOLD CV: 0.0375901�0.0019
    """
    best_parameters_tf_idf_true = {'FUSION': 0.11932917388021072, 'ItemDotCF': 0.714527859515967,
                                   'ItemCBF_CF': 0.314038888909047, 'RP3BETA_SIDE': 0.1419501369179094}

    best_parameters = {'FUSION': 0.027727001572924077, 'ItemDotCF': 0.5547085458866265,
                       'ItemCBF_CF': 0.3781014922999707,
                       'RP3BETA_SIDE': 0.34790282048827864}

    @classmethod
    def get_model(cls, URM_train, ICM_all):
        from src.model.HybridRecommender.HybridWeightedAverageRecommender import HybridWeightedAverageRecommender

        fusion = FusionMergeItem_CBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)
        item_cbf_cf = ItemCBF_CF.get_model(URM_train=URM_train, ICM_train=ICM_all, load_model=False)
        rp3beta = RP3Beta_side_info.get_model(URM_train=URM_train, ICM_train=ICM_all, apply_tf_idf=False)
        item_dot = ItemDotCF.get_model(URM_train=URM_train, load_model=False)

        hybrid = HybridWeightedAverageRecommender(URM_train, normalize=True)
        hybrid.add_fitted_model("ItemDotCF", item_dot)
        hybrid.add_fitted_model("FUSION", fusion)
        hybrid.add_fitted_model("ItemCBF_CF", item_cbf_cf)
        hybrid.add_fitted_model("RP3BETA_SIDE", rp3beta)

        hybrid.fit(**cls.get_best_parameters())
        return hybrid


class ItemDotCF(ICollaborativeModel):
    """
    - MAP 10 FOLD: 0.0352704�0.0015

    Other parameters MAP 10 FOLD:
    MAP: 0.0344132�0.0017 = {'topK': 10, 'shrink': 766, 'normalize': True, 'feature_weighting': 'none'}
    MAP: 0.0350459�0.0014 = {'topK': 5, 'shrink': 620, 'normalize': True, 'feature_weighting': 'none'}
    """
    from src.model.KNN.ItemKNNDotCFRecommender import ItemKNNDotCFRecommender
    best_parameters = {'topK': 3, 'shrink': 2000, 'normalize': True, 'feature_weighting': 'none'}
    recommender_class = ItemKNNDotCFRecommender
    recommender_name = "ItemDotCF"


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


class NewPureSVD_side_info(IBestModel):
    """
    MAP 5CV: 0.028324�0.0018
    MAP 10CV TF IDF True: 0.0285517�0.0027
    MAP 10CV TF IDF FALSE: 0.0272885�0.0026
    """
    best_parameters = {'num_factors': 496, 'n_oversamples': 2, 'n_iter': 20, 'feature_weighting': 'TF-IDF'}

    @classmethod
    def get_model(cls, URM_train, ICM_train, apply_tf_idf=True):
        from src.model.MatrixFactorization.NewPureSVDRecommender import NewPureSVDRecommender
        from course_lib.Base.IR_feature_weighting import TF_IDF
        if apply_tf_idf:
            URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        else:
            URM_train_side_info = sps.vstack([URM_train, ICM_train.T]).tocsr()
        model = NewPureSVDRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class PureSVD_side_info(IBestModel):
    """
    PureSVD recommender with side info by using TF_IDF([URM_train, ICM_all.T])
    - MAP 5 FOLD: 0.027078�0.0019
    - MAP 10 FOLD: MAP: 0.0265599�0.0025
    - MAP 10 FOLD TF_IDF_FALSE: MAP: 0.0212042�0.0023
    """
    best_parameters = {'num_factors': 435}

    @classmethod
    def get_model(cls, URM_train, ICM_train, apply_tf_idf=True):
        from course_lib.MatrixFactorization.PureSVDRecommender import PureSVDRecommender
        from course_lib.Base.IR_feature_weighting import TF_IDF
        if apply_tf_idf:
            URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        else:
            URM_train_side_info = sps.vstack([URM_train, ICM_train.T]).tocsr()
        model = PureSVDRecommender(URM_train_side_info)
        model.fit(**cls.get_best_parameters())
        return model


class RP3Beta_side_info(IBestModel):
    """
    RP3 beta with side info by using TF_IDF([URM_train, ICM_all.T])

    - MAP 5 FOLD: 0.035434�0.0013
    - MAP 10 FOLD: 0.0328794�0.0017
    - MAP 10 FOLD APPLY TF IDF FALSE: 0.0351791�0.0018
    """

    best_parameters = {'topK': 28, 'alpha': 0.008124745090408949,
                       'beta': 0.0051792301071096345, 'normalize_similarity': True}

    @classmethod
    def get_model(cls, URM_train, ICM_train, apply_tf_idf=False):
        from course_lib.Base.IR_feature_weighting import TF_IDF

        if apply_tf_idf:
            URM_train_side_info = TF_IDF(sps.vstack([URM_train, ICM_train.T])).tocsr()
        else:
            URM_train_side_info = sps.vstack([URM_train, ICM_train.T]).tocsr()

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

    MAP 10 FOLD LT 23 OF THIS MODEL: 0.0096621�0.0009

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


class FusionMergeItem_CBF_CF(IBestModel):
    """
    Fusion, i.e. bagging w/o bootstrap, merge of best models: Item_CBF_CF using ICM_all (sub_class, price, asset,
    sub_class_count and item_pop; all discretized with bins 200, 200, 50, 50)
     - MAP-K1 CV10 (lt 22): 0.0365217±0.0023
    """
    best_parameters = {'num_models': 10, 'topK': 16}
    recommender_name = "FusionMergeItem_CBF_CF"

    @classmethod
    def get_hyperparameters(cls):
        hyper_parameters_range = {}
        for par, value in ItemCBF_CF.get_best_parameters().items():
            hyper_parameters_range[par] = Categorical([value])
        hyper_parameters_range['topK'] = Integer(low=3, high=30)
        hyper_parameters_range['shrink'] = Integer(low=1000, high=2000)
        return hyper_parameters_range

    @classmethod
    def get_model(cls, URM_train, ICM_train, load_model=False, save_model=False):
        from src.model.Ensemble.BaggingMergeRecommender import BaggingMergeItemSimilarityRecommender
        from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender
        model = BaggingMergeItemSimilarityRecommender(URM_train, ItemKNNCBFCFRecommender, do_bootstrap=False,
                                                      ICM_train=ICM_train)

        try:
            if load_model:
                model = cls._load_model(model)
                return model
        except FileNotFoundError:
            print("WARNING: Cannot find model to be loaded")

        model.fit(topK=16, num_models=10, hyper_parameters_range=cls.get_hyperparameters())
        if save_model:
            cls._save_model(model)
        return model
