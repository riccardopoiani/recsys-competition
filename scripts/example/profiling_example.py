import cProfile, pstats, io
from src.data_management.RecSys2018Reader import RecSys2018Reader
from course_lib.Data_manager.DataSplitter_k_fold import DataSplitter_Warm_k_fold
from course_lib.KNN.UserKNNCFRecommender import UserKNNCFRecommender
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.model.HybridRankBasedRecommender import HybridRankBasedRecommender

if __name__ == '__main__':
    dataset = RecSys2018Reader("../data/train.csv", "../data/tracks.csv")
    dataset = DataSplitter_Warm_k_fold(dataset, n_folds=10)
    dataset.load_data()

    URM_train, URM_test = dataset.get_URM_train_for_test_fold(n_test_fold=9)

    user_CF_model = UserKNNCFRecommender(URM_train)
    user_CF_model.fit(topK=177, shrink=4, feature_weighting="TF-IDF")

    item_CF_model = ItemKNNCFRecommender(URM_train)
    item_CF_model.fit(topK=548, shrink=447, feature_weighting="TF-IDF")

    hybrid_model = HybridRankBasedRecommender(URM_train, multiplier_cutoff=5)
    hybrid_model.add_fitted_model("USER_CF", user_CF_model)
    hybrid_model.add_fitted_model("ITEM_CF", item_CF_model)
    hybrid_model.fit(USER_CF=0.5, ITEM_CF=0.5)

    pr = cProfile.Profile()
    pr.enable()

    for user in range(10000):
        hybrid_model.recommend(user, cutoff=10)

    pr.disable()
    s = io.StringIO()
    sortby = "cumulative"
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())