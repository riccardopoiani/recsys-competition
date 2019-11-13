from src.data_management.RecSys2018Reader import RecSys2018Reader
from src.model_management.NewEvaluator import *
from course_lib.KNN.ItemKNNCFRecommender import *

if __name__ == '__main__':
    dataset = RecSys2018Reader("../data/train.csv", "../data/tracks.csv")

    cutoff = 10
    cross_valuator = EvaluatorCrossValidation(dataset, cutoff, n_folds=3)
    recommender_args = []
    recommender_keywargs = {'topK': 548,'shrink': 447,'similarity': 'cosine','normalize': True,
                            'feature_weighting': 'TF-IDF'}

    results = cross_valuator.crossevaluateRecommender(ItemKNNCFRecommender, **recommender_keywargs)

    print(results)
