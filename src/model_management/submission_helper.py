def write_submission_file(recommender, path, userlist):
    '''
    :param recommender: tells how to recommend item for the users. It must be a
    recommender class
    :param path: where to store the model
    :param userlist: list of users to recommend. The list get sorted in the function
    so you don't need to do it before
    :return: none
    '''
    from tqdm import tqdm

    userlist.sort()
    f = open(path, "w+")
    f.write("playlist_id,track_ids\n")
    for i in tqdm(range(len(userlist)), mininterval=10, maxinterval=30):
        f.write(str(userlist[i]))
        f.write(",")
        recommendation_list = recommender.recommend(userlist[i], cutoff=10)
        for item in recommendation_list:
            f.write(str(item))
            f.write(" ")
        f.write("\n")
    f.close()