from src.data_management.RecSys2018Reader import RecSys2018Reader
from course_lib.KNN.ItemKNNCFRecommender import ItemKNNCFRecommender
from src.model_management.submission_helper import *
from src.data_management.data_reader import read_target_playlist

if __name__ == '__main__':
    # Data loading
    dataset = RecSys2018Reader("../data/train.csv", "../data/tracks.csv")
    dataset.load_data()
    URM_all = dataset.get_URM_all()

    # Model building
    item_cf_keywargs = {'topK': 548, 'shrink': 447, 'similarity': 'cosine', 'normalize': True, 'feature_weighting': 'TF-IDF'}
    item_cf = ItemKNNCFRecommender(URM_train=URM_all)
    item_cf.fit(**item_cf_keywargs)

    # Getting target tracklist
    target_playlist = read_target_playlist()

    print(type(target_playlist))
    print(target_playlist)

    write_submission_file(path="../report/submitted_models/item_cf_10_map_fixed.csv", userlist=target_playlist, recommender=item_cf)


