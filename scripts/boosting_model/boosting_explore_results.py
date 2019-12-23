import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xgboost as xgb
import os

from datetime import datetime
from src.data_management.New_DataSplitter_leave_k_out import New_DataSplitter_leave_k_out
from src.data_management.RecSys2019Reader import RecSys2019Reader
from src.data_management.data_reader import get_ICM_train, get_UCM_train
from src.model import new_best_models
from src.model.Ensemble.Boosting.boosting_preprocessing import add_label, preprocess_dataframe_after_reading
from src.utils.general_utility_functions import get_split_seed


def plot_score_distribution(column_name):
    # Let's check train and validation performances, min-max standardized
    train_mxd = train_df[column_name].values
    valid_mxd = valid_df[column_name].values

    # Sort them
    train_mxd = np.sort(train_mxd)
    valid_mxd = np.sort(valid_mxd)

    # Plot the distributions
    plt.title('Distributions of {} scores between the two datasets'.format(column_name))
    plt.xlabel('(user, item) index')
    plt.ylabel('Score')
    plt.plot(train_mxd, label="Training")
    plt.plot(valid_mxd, label="Validation")
    plt.legend()

    fig_t = plt.gcf()
    fig_t.show()
    new_file_t = output_folder_path + "scores_" + column_name + ".png"
    fig_t.savefig(new_file_t)


def plot_score_distribution_unsorted(column_name_list, train=True):
    # Let's check train and validation performances, min-max standardized
    if train:
        df = train_df
    else:
        df = valid_df

    # Plot the distributions
    plt.title("Distribution of scores unsorted")
    plt.xlabel('(user, item) index')
    plt.ylabel('Score')
    for name in column_name_list:
        plt.plot(df[name].values, label=name)
    plt.legend()

    fig_t = plt.gcf()
    fig_t.show()
    new_file_t = output_folder_path + "score_unsorted.png"
    fig_t.savefig(new_file_t)


if __name__ == '__main__':
    # Initial settings
    show_score_distribution = True
    print_mean = True
    model_path = "../../report/hp_tuning/boosting/Dec23_11-22-35_k_out_value_3_eval/best_model6"
    print_example_power_user = True
    show_extreme_scores = True
    show_mean_scores = True
    show_power_user_not_extreme = True
    show_power_user_not_extreme_sorted = True
    show_middle_mean = True

    # Path creation
    version_path = "../../report/graphics/boosting/"
    now = datetime.now().strftime('%b%d_%H-%M-%S')
    now = now + "k3_foh_5_v2/"
    output_folder_path = version_path + now
    output_file_name = output_folder_path + "results.txt"
    try:
        if not os.path.exists(output_folder_path):
            os.mkdir(output_folder_path)
    except FileNotFoundError as e:
        os.makedirs(output_folder_path)

    f = open(output_file_name, "w")

    # Data loading
    root_data_path = "../../data/"
    data_reader = RecSys2019Reader(root_data_path)
    data_reader = New_DataSplitter_leave_k_out(data_reader, k_out_value=3, use_validation_set=False,
                                               force_new_split=True, seed=get_split_seed())
    data_reader.load_data()
    URM_train, URM_test = data_reader.get_holdout_split()
    ICM_all = get_ICM_train(data_reader)
    UCM_all = get_UCM_train(data_reader)
    dataframe_path = "../../boosting_dataframe/"
    train_df = pd.read_csv(dataframe_path + "train_df_20_advanced_foh_5.csv")
    valid_df = pd.read_csv(dataframe_path + "valid_df_20_advanced_foh_5.csv")
    train_df = preprocess_dataframe_after_reading(train_df)
    train_df_with_labels = train_df.copy()
    train_df = train_df.drop(columns=["label"], inplace=False)
    valid_df = preprocess_dataframe_after_reading(valid_df)
    print("Retrieving training labels...", end="")
    y_train, non_zero_count, total = add_label(data_frame=train_df, URM_train=URM_train)
    print("Done")

    boosting = new_best_models.BoostingFoh5.get_model(URM_train=URM_train, train_df=train_df, y_train=y_train,
                                                      valid_df=valid_df,
                                                      model_path=model_path)

    boosting.RECOMMENDER_NAME = "BOOSTING"

    # Setting additional dataframes
    y_train_valid, non_zero_count_vaid, total_valid = add_label(data_frame=valid_df, URM_train=URM_test)
    valid_df['label'] = y_train_valid
    train_df = train_df_with_labels

    power_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 40
    power_users = np.arange(URM_train.shape[0])[power_users_mask]

    rare_users = np.ediff1d(URM_train.tocsr().indptr) < 7
    rare_users = np.arange(URM_train.shape[0])[rare_users]

    total_users = np.arange(URM_train.shape[0])
    middle_users = np.in1d(total_users, np.unique(np.concatenate((power_users, rare_users))), invert=True)
    middle_users = total_users[middle_users]

    power_train_df = train_df[train_df['user_id'].isin(power_users)]
    power_valid_df = valid_df[valid_df['user_id'].isin(power_users)]

    rare_train_df = train_df[train_df['user_id'].isin(rare_users)]
    rare_valid_df = valid_df[valid_df['user_id'].isin(rare_users)]

    middle_train_df = train_df[train_df['user_id'].isin(middle_users)]
    middle_valid_df = valid_df[valid_df['user_id'].isin(middle_users)]

    extreme_users_mask = np.ediff1d(URM_train.tocsr().indptr) > 100
    extreme_users = np.arange(URM_train.shape[0])[extreme_users_mask]

    extreme_train_df = train_df[train_df['user_id'].isin(extreme_users)]
    extreme_valid_df = valid_df[valid_df['user_id'].isin(extreme_users)]

    power_users_not_extreme = np.in1d(power_users, extreme_users, invert=True)
    power_users_not_extreme = power_users[power_users_not_extreme]

    # Plotting scores distribution
    if show_score_distribution:
        plot_score_distribution("MixedItem")
        plot_score_distribution("ItemCBF_CF")
        plot_score_distribution("RP3BetaSideInfo")
        plot_score_distribution("UserCF")
        plot_score_distribution("ItemCBF_all_FW")

        name_list = ["MixedItem", "UserCF"]
        plot_score_distribution_unsorted(name_list)
        plot_score_distribution_unsorted(name_list, train=False)

    # Scores + Label exploration
    if print_mean:
        print("Training dataframe \n")
        print("Train dataframe label mean {} \n".format(train_df['label'].mean()))
        print("Power (>40) train dataframe label mean {} \n".format(power_train_df['label'].mean()))
        print("Rare (<7) train dataframe label mean {} \n".format(rare_train_df['label'].mean()))
        print("Middle (>=7 <=40) train dataframe label mean {} \n".format(
            middle_train_df['label'].mean()))
        print("Extreme (>100) train dataframe label mean {} \n".format(extreme_train_df['label'].mean()))

        print("\nValidation dataframe \n")
        print("Valid dataframe label mean {} \n".format(valid_df['label'].mean()))
        print("Power (>40) valid dataframe label mean {} \n".format(power_valid_df['label'].mean()))
        print("Rare (<7) valid dataframe label mean {} \n".format(rare_valid_df['label'].mean()))
        print("Middle (>=7 <=40) valid dataframe label mean {} \n".format(
            middle_valid_df['label'].mean()))

        print("Extreme (>100) valid dataframe label mean {} \n".format(extreme_valid_df['label'].mean()))
        print("\n")

        f.write("Training dataframe \n")
        f.write("Train dataframe label mean {} \n".format(train_df['label'].mean()))
        f.write("Power (>40) train dataframe label mean {} \n".format(power_train_df['label'].mean()))
        f.write("Rare (<7) train dataframe label mean {} \n".format(rare_train_df['label'].mean()))
        f.write("Middle (>=7 <=40) train dataframe label mean {} \n".format(
            middle_train_df['label'].mean()))
        f.write("Extreme (>100) train dataframe label mean {} \n".format(extreme_train_df['label'].mean()))

        f.write("\nValidation dataframe \n")
        f.write("Valid dataframe label mean {} \n".format(valid_df['label'].mean()))
        f.write("Power (>40) valid dataframe label mean {} \n".format(power_valid_df['label'].mean()))
        f.write("Rare (<7) valid dataframe label mean {} \n".format(rare_valid_df['label'].mean()))
        f.write("Middle (>=7 <=40) valid dataframe label mean {} \n".format(
            middle_valid_df['label'].mean()))

        f.write("Extreme (>100) valid dataframe label mean {} \n".format(extreme_valid_df['label'].mean()))
        f.write("\n")

    if print_example_power_user:
        print("Example of extreme user \n")
        f.write("Example of extreme user \n")
        temp_user = extreme_users[0]
        user_example_df = valid_df[valid_df['user_id'] == temp_user]
        user_example_df = user_example_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
        print("Predictions")
        f.write("Predictions\n")
        print(boosting.bst.predict(xgb.DMatrix(user_example_df)))
        f.write(str(boosting.bst.predict(xgb.DMatrix(user_example_df)).tolist()))

    if show_extreme_scores:
        mean_extreme = []
        for user in extreme_users:
            curr_user_valid_df = valid_df[valid_df['user_id'] == user].copy()
            curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
            res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
            mean_extreme.append(res.mean())
            print("User {}: media {} - std {} - activity {}".format(user, res.mean(), res.std(),
                                                                    URM_train[user].indices.size))

        plt.title("Distribution of mean predictions for extreme users")
        plt.xlabel('power user index')
        plt.ylabel('Mean prediction')
        plt.plot(mean_extreme)
        fig = plt.gcf()
        fig.show()
        new_file = output_folder_path + "extreme_unsort.png"
        fig.savefig(new_file)

        plt.title("Sorted Distribution of mean predictions for extreme users")
        plt.xlabel('power user index')
        plt.ylabel('Mean prediction')
        plt.plot(np.sort(np.array(mean_extreme)))
        fig = plt.gcf()
        fig.show()
        new_file = output_folder_path + "extreme_sort.png"
        fig.savefig(new_file)

    if show_mean_scores:
        mean_middle = []
        for user in middle_users:
            curr_user_valid_df = valid_df[valid_df['user_id'] == user].copy()
            curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
            res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
            mean_middle.append(res.mean())

        plt.title("Distribution of mean predictions for middle users")
        plt.xlabel('Middle user index')
        plt.ylabel('Mean predictions')
        plt.plot(mean_middle)
        fig = plt.gcf()
        fig.show()
        new_file = output_folder_path + "middle_unsort.png"
        fig.savefig(new_file)

        plt.title("Sorted Distribution of mean predictions for middle users")
        plt.xlabel('Middle user index')
        plt.ylabel('Mean predictions')
        plt.plot(np.sort(np.array(mean_middle)))
        fig = plt.gcf()
        fig.show()
        new_file = output_folder_path + "middle_sort.png"
        fig.savefig(new_file)

    if show_power_user_not_extreme:
        mean_power_not_extreme = []
        for user in power_users_not_extreme:
            curr_user_valid_df = valid_df[valid_df['user_id'] == user].copy()
            curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
            res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
            mean_power_not_extreme.append(res.mean())

        plt.title("Distribution of mean predictions for power users not extreme")
        plt.xlabel('User index')
        plt.ylabel('Mean predictions')
        plt.plot(mean_power_not_extreme)
        fig = plt.gcf()
        fig.show()
        new_file = output_folder_path + "power_next_unsort.png"
        fig.savefig(new_file)

        plt.title("Distribution of mean predictions for power users not extreme")
        plt.xlabel('User index')
        plt.ylabel('Mean predictions')
        plt.plot(np.sort(np.array(mean_power_not_extreme)))
        fig = plt.gcf()
        fig.show()
        new_file = output_folder_path + "power_next_sort.png"
        fig.savefig(new_file)

        if show_power_user_not_extreme_sorted:
            act_pune = URM_train[power_users_not_extreme].sum(axis=1)
            act_pune = np.argsort(np.array(act_pune).squeeze())
            power_users_not_extreme = power_users_not_extreme[act_pune]

            mean_power_not_extreme_sorted = []
            for user in power_users_not_extreme:
                curr_user_valid_df = valid_df[valid_df['user_id'] == user].copy()
                curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
                res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
                mean_power_not_extreme_sorted.append(res.mean())

            plt.title("Distribution of mean predictions for power users not extreme")
            plt.xlabel('User index')
            plt.ylabel('Mean predictions')
            plt.plot(mean_power_not_extreme_sorted, label="Sorted By user act")
            plt.plot(np.sort(np.array(mean_power_not_extreme)), label="sort by mean")
            plt.legend()
            fig = plt.gcf()
            fig.show()
            new_file = output_folder_path + "power_next_act.png"
            fig.savefig(new_file)

    if show_middle_mean:
        middle_high_mean = []
        t = 0.8
        for user in middle_users:
            curr_user_valid_df = valid_df[valid_df['user_id'] == user].copy()
            curr_user_valid_df = curr_user_valid_df.drop(columns=['user_id', 'item_id', 'label'], inplace=False)
            res = boosting.bst.predict(xgb.DMatrix(curr_user_valid_df))
            if res.mean() > 0.8:
                middle_high_mean.append(user)

        print("Middle users size {}\n".format(middle_users.size))
        print("Middle high mean size {}\n".format(len(middle_high_mean)))
        f.write("Middle users size {}\n".format(middle_users.size))
        f.write("Middle high mean size {}\n".format(len(middle_high_mean)))

    f.close()
