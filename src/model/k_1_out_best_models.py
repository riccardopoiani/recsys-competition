from src.model.Interface import ICollaborativeModel, IContentModel, IBestModel


# ----- ABBREVIATIONS ------ #
# K1: keep-1-out
# CV: cross validation

class ItemCBF_CF(IContentModel):
    """
    Item CF
     - MAP@10 K1-CV5 (only warm and target):
    """
    from src.model.KNN.ItemKNNCBFCFRecommender import ItemKNNCBFCFRecommender

    best_parameters = {'topK': 17, 'shrink': 1463, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 0.07899555402911075, 'feature_weighting': 'TF-IDF'}
    recommender_class = ItemKNNCBFCFRecommender
    recommender_name = "ItemCBF_CF"


class P3Alpha(ICollaborativeModel):
    """
    P3Alpha
     - MAP@10 K1-CV (only warm): 0.0463
     - MAP@10 K1-CV (only warm and target): TODO
    """
    from course_lib.GraphBased.P3alphaRecommender import P3alphaRecommender

    best_parameters = {'topK': 122, 'alpha': 0.38923114168898876, 'normalize_similarity': True}
    recommender_class = P3alphaRecommender
    recommender_name = "P3alpha"


class HybridNormWeightedAvgAll(IBestModel):
    """
    Hybrid of Normalized weighted average ranking of almost all models

     - MAP@10 K1-10CV (only warm) with ICM_all_weighted and UCM_all:
    """
    best_parameters = {'strategy': 'norm_weighted_avg', 'multiplier_cutoff': 2, 'WEIGHTED_AVG_ITEM': 0.9679497374745649,
                       'S_PURE_SVD': 0.02023761457683704, 'S_IALS': 0.007225989992151629,
                       'USER_CBF_CF': 0.05179513388991243,
                       'USER_CF': 0.03061248068550649}

    @classmethod
    def _get_all_models(cls, URM_train, ICM_all, UCM_all):
        from src.model import new_best_models

        all_models = {'WEIGHTED_AVG_ITEM': new_best_models.WeightedAverageItemBased.get_model(URM_train, ICM_all),
                      'S_PURE_SVD': new_best_models.PureSVDSideInfo.get_model(URM_train, ICM_all),
                      'S_IALS': new_best_models.IALSSideInfo.get_model(URM_train, ICM_all),
                      'USER_CBF_CF': new_best_models.UserCBF_CF_Warm.get_model(URM_train, UCM_all),
                      'USER_CF': new_best_models.UserCF.get_model(URM_train)}

        return all_models

    @classmethod
    def get_model(cls, URM_train, ICM_train, UCM_train):
        from src.model.HybridRecommender.HybridRankBasedRecommender import HybridRankBasedRecommender

        all_models = cls._get_all_models(URM_train, ICM_train, UCM_train)
        hybrid = HybridRankBasedRecommender(URM_train)
        for model_name, model_object in all_models.items():
            hybrid.add_fitted_model(model_name, model_object)
        hybrid.fit(**cls.get_best_parameters())

        return hybrid


class UserCBF_Cold(IBestModel):
    """
    User CBF tuned with URM_train and UCM (containing age, region, user_act)
     - MAP on tuning (k_out_3 and testing on 2, 3, 4 len users): 0.0117
     - MAP on cold users (k_1_out): 0.01735
    """
    best_parameters = {'topK': 3372, 'shrink': 1086, 'similarity': 'asymmetric', 'normalize': True,
                       'asymmetric_alpha': 1.5033071260303803, 'feature_weighting': 'BM25',
                       'interactions_feature_weighting': 'BM25'}

    @classmethod
    def get_model(cls, URM_train, UCM_train):
        from src.model.KNN.UserKNNCBFRecommender import UserKNNCBFRecommender
        model = UserKNNCBFRecommender(URM_train=URM_train, UCM_train=UCM_train)
        model.fit(**cls.get_best_parameters())
        return model


class HybridDemographicWithLT23AndUT22(IBestModel):
    """
    Final hybrid model composed by two hybrid: one for smaller profile len users and one for bigger profile len users

     - The threshold is set heuristically (I have tested threshold 20 and threshold 26, but there are not much changes
     in the MAP)
     - UCM_train is the one from get_UCM_train_new or get_UCM_all_new
     - ICM_train is the one from get_ICM_train_new or get_ICM_all_new
    """

    threshold = 23

    @classmethod
    def get_model(cls, URM_train, ICM_train, UCM_train):
        from src.model.HybridRecommender.HybridDemographicRecommender import HybridDemographicRecommender
        from src.model import best_models_lower_threshold_23, best_models_upper_threshold_22
        import numpy as np

        lt_23_recommender = best_models_lower_threshold_23.WeightedAverageItemBasedWithRP3.get_model(URM_train,
                                                                                                     ICM_train)
        ut_22_recommender = best_models_upper_threshold_22.WeightedAverageAll.get_model(URM_train, ICM_train, UCM_train)
        lt_23_users_mask = np.ediff1d(URM_train.tocsr().indptr) >= cls.threshold
        lt_23_users = np.arange(URM_train.shape[0])[lt_23_users_mask]
        ut_23_users = np.arange(URM_train.shape[0])[~lt_23_users_mask]

        model = HybridDemographicRecommender(URM_train=URM_train)
        model.add_user_group(1, lt_23_users)
        model.add_user_group(2, ut_23_users)
        model.add_relation_recommender_group(lt_23_recommender, 1)
        model.add_relation_recommender_group(ut_22_recommender, 2)
        model.fit()

        return model
