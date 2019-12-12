import numpy as np

from course_lib.Base.BaseRecommender import BaseRecommender
from src.model.Ensemble.Boosting.Boosting import BoostingFixedData
from src.model.Ensemble.Boosting.boosting_preprocessing import get_dataframe, add_label
from course_lib.Base.Evaluation.metrics import MAP
import scipy.sparse as sps


def MAP(is_relevant, relevant_items):
    if relevant_items.size == 0:
        return 0

    # Cumulative sum: precision at 1, at 2, at 3 ...
    p_at_k = is_relevant * np.cumsum(is_relevant, dtype=np.float32) / (1 + np.arange(is_relevant.shape[0]))

    map_score = np.sum(p_at_k) / np.min([relevant_items.shape[0], is_relevant.shape[0]])

    return map_score


def compute_map_at_10(recommendations, URM_test, users_to_validate):
    cumulative_MAP = 0.0

    num_eval = 0
    URM_test = sps.csr_matrix(URM_test)
    n_users = URM_test.shape[0]

    for i, user_id in enumerate(users_to_validate):

        if user_id % 10000 == 0:
            print("Evaluated user {} of {}".format(user_id, n_users))

        start_pos = URM_test.indptr[user_id]
        end_pos = URM_test.indptr[user_id + 1]

        if end_pos - start_pos > 0:
            relevant_items = URM_test.indices[start_pos:end_pos]

            recommended_items = recommendations[i]
            num_eval += 1

            is_relevant = np.in1d(recommended_items, relevant_items, assume_unique=True)

            cumulative_MAP += MAP(is_relevant, relevant_items)

    cumulative_MAP /= num_eval

    return cumulative_MAP


def sample_parameters(parameters: dict):
    keys = list(parameters.keys())
    sample = {}
    for k in keys:
        values = parameters[k]
        n_param = len(values)
        if n_param == 1:
            sample[k] = values[1]
        else:
            idx = np.random.randint(low=0, high=n_param)
            sample[k] = values[idx]

    return sample


def run_xgb_tuning(user_to_validate, URM_train, URM_test, main_recommender: BaseRecommender, recommender_list: list,
                   mapper: dict,
                   n_trials=40,
                   max_iter_per_trial=5000, n_early_stopping=15,
                   output_folder_file="",
                   objective="binary:logistic", parameters=None,
                   cutoff=20, data_path="../../data/",
                   valid_size=0.2,
                   best_model_folder=""):
    # Retrieve data for boosting
    train_df = get_dataframe(user_id_array=user_to_validate,
                             remove_seen_flag=False,
                             cutoff=cutoff,
                             main_recommender=main_recommender,
                             recommender_list=recommender_list,
                             mapper=mapper, URM_train=URM_train, path=data_path)
    y_train, non_zero_count, total = add_label(data_frame=train_df, URM_train=URM_train)
    valid_df = get_dataframe(user_id_array=user_to_validate,
                             remove_seen_flag=True,
                             cutoff=cutoff,
                             main_recommender=main_recommender,
                             recommender_list=recommender_list,
                             mapper=mapper,
                             URM_train=URM_train,
                             path=data_path)

    scale_pos_weight = (total - non_zero_count) / non_zero_count

    if parameters is None:
        parameters = {"learning_rate": [0.1, 0.01, 0.001],
                      "gamma": [0.001, 0.1, 0.3, 0.5, 0.8],  # min loss required to split a leaf
                      "max_depth": [2, 4, 7],  # the larger, the higher prob. to overfitting
                      "max_delta_step": [0, 1, 5, 10],  # needed for unbalanced dataset
                      "subsample": [0.2, 0.4, 0.5, 0.6, 0.7],  # sub-sampling of data before growing trees
                      "colsample_bytree": [0.3, 0.6, 0.8, 1.0],  # sub-sampling of columns
                      "scale_pos_weight": [scale_pos_weight],  # to deal with unbalanced dataset
                      "objective": [objective]  # Objective function to be optimized
                      }

    f = open(output_folder_file, "w")
    f.write("Tuning XGBoost \n")
    f.write("Parameters: \n " + str(parameters))
    f.write("\n N_trials: {} \n".format(n_trials))
    f.write("Max iteration per trial: {}\n".format(max_iter_per_trial))
    f.write("Early stopping every {} iterations\n".format(n_early_stopping))

    f.write("\n\n Begin tuning \n\n")

    max_map = -1
    best_param = {}
    best_trial = -1

    for i in range(n_trials):
        sample = sample_parameters(parameters)
        print("Trying configuration: " + str(sample))
        f.write(str(sample))

        boosting = BoostingFixedData(URM_train=URM_train, X=train_df, y=y_train, df_test=valid_df,
                                     cutoff=cutoff, valid_size=valid_size)

        boosting.train(num_round=max_iter_per_trial, param=sample, early_stopping_round=n_early_stopping)

        predictions = boosting.recommend(user_id_array=user_to_validate)

        map_10 = compute_map_at_10(predictions, URM_test, user_to_validate)
        if map_10 > max_map:
            max_map = map_10
            best_param = sample
            best_trial = i
            print("Saving best model...", end="")
            boosting.bst.save_model(best_model_folder + "best_model{}".format(i))
            print("Done")

        f.write("Map@10 {}".format(map_10))

    # Best results
    f.write("\n\n")
    f.write("Best MAP score: {}".format(max_map))
    f.write("Best config " + str(best_param))
    f.write("Best trial " + str(best_trial))
    f.close()
