from course_lib.Base.BaseRecommender import BaseRecommender


def write_submission_file(recommender, path, userlist):
    """
    :param recommender: tells how to recommend item for the users. It must be a
    recommender class
    :param path: where to store the model
    :param userlist: list of users to recommend. The list get sorted in the function
    so you don't need to do it before
    :return: none
    """
    from tqdm import tqdm

    userlist.sort()
    f = open(path, "w+")
    f.write("user_id,item_list\n")
    for i in tqdm(range(len(userlist)), mininterval=10, maxinterval=30):
        f.write(str(userlist[i]))
        f.write(",")
        recommendation_list = recommender.recommend(userlist[i], cutoff=10)
        for item in recommendation_list:
            f.write(str(item))
            f.write(" ")
        f.write("\n")
    f.close()


def write_submission_file_batch(recommender: BaseRecommender, path, userlist, batches=10):
    """
    :param recommender: tells how to recommend item for the users. It must be a
    recommender class
    :param path: where to store the model
    :param userlist: list of users to recommend. The list get sorted in the function
    so you don't need to do it before
    :param batches: number of epochs to recommend user list
    :return: none
    """
    from tqdm import tqdm
    import numpy as np

    f = open(path, "w+")
    f.write("user_id,item_list\n")
    user_split_lists = np.array_split(userlist, batches)
    for i in tqdm(range(len(user_split_lists))):
        users = user_split_lists[i]
        recommendation_list = recommender.recommend(users, cutoff=10, remove_seen_flag=True)
        for j in range(len(recommendation_list)):
            f.write(str(users[j]))
            f.write(",")
            for item in recommendation_list[j]:
                f.write(str(item))
                f.write(" ")
            f.write("\n")
    f.close()
