from src.model_management.model_result_reader import *
from src.plots.plot_evaluation_helper import *
from src.model_management.evaluator import *
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
import numpy as np
from datetime import datetime

def basic_plots_from_tuning_results(path, recommender_class, URM_train, URM_test,
                                    metric='MAP', save_on_file=False, retrain_model=True,
                                    is_compare_top_pop=True,
                                    compare_top_pop_points=None,
                                    demographic_list_name=None, demographic_list=None, output_path_folder="",
                                    ICM = None):
    '''
    Plot some graphics concerning the results of hyperparameter procedure.
    In particular, first samples are plotted (where the bayesian method have searched for solutions).
    Then, it searches for the best model found, and, do some plots about it.
    In particular:
    - Recommendation distribution
    - Comparison against top popular recommender
    - Comparison with user activity
    - Other comparison against an arbitrary number of demographic to be specified

    :param path: path to the folder of the hp results
    :param recommender_class: recommender class subject to tuning
    :param URM_train: URM train on which to train the recommender
    :param URM_test: URM test on which to test the recommender
    :param metric: specify how to search for the best model
    :param save_on_file: if you want to save graphics on file
    :param retrain_model: if model has to be re-trained (only option so far)
    :param is_compare_top_pop: if you have to perform the comparison with a top popular recommender
    :param compare_top_pop_points: comparison with the top popular with this points
    :param demographic_list_name: list of addiontal demographics name
    :param demographic_list: list of additional demographics
    :param output_path_folder: where image should be saved, if specified
    :return: None
    '''
    results = read_folder_metadata(path)

    # Create folder if it does not exist
    if save_on_file:
        try:
            if not os.path.exists(output_path_folder):
                os.mkdir(output_path_folder)
        except FileNotFoundError as e:
            os.makedirs(output_path_folder)

    if compare_top_pop_points is None:
        compare_top_pop_points = [10, 100, 500, 1000]

    # PLOT WHERE SAMPLES HAVE BEEN TAKEN
    for count, r in enumerate(results):
        arhr_index = -1
        for j in range(0, len(r.columns)):
            if r.columns[j] == 'ARHR':
                arhr_index = j
                break
        dimension_list = r.columns[0:arhr_index]

        # Plot
        fig, ax = plot_sample_evaluations(r, dimension_list)
        fig.show()

        # Save the plot on file
        if save_on_file:
            output_path_file = output_path_folder + "sample_distr_" + str(count) + ".png"
            fig.savefig(output_path_file, bbox_inches='tight')
            print("Add feature of saving on files the plots")

    # MAX FOUND SO FAR
    t = results[0][metric].nlargest(1)
    max_so_far = t.values[0]
    best_r = results[0]
    best_idx = t.index[0]
    for i in range(1, len(results)):
        t = results[i][metric].nlargest(1)
        if t.values > max_so_far:
            max_so_far = t.values[0]
            best_r = results[i]
            best_idx = t.index[0]

    best_df_row = best_r.loc[best_idx:best_idx]

    # Finding parameter of the best model
    arhr_index = -1
    for j in range(0, len(best_r.columns)):
        if best_df_row.columns[j] == 'ARHR':
            arhr_index = j
            break
    dimension_list = best_df_row.columns[0:arhr_index]
    method_param_df = best_df_row[dimension_list]
    keywargs = {}
    for d in dimension_list:
        keywargs[d] = method_param_df[d].values[0]

    print("Parameter of the best model")
    print(keywargs)
    print("Best model result")
    print(best_df_row)

    if retrain_model:
        if ICM is not None:
            recommender_instance = recommender_class(ICM, URM_train)
        else:
            recommender_instance = recommender_class(URM_train)
        recommender_instance.fit(**keywargs)
    else:
        print("Read the best model from file")
        raise NotImplemented("Not implemented feature")

    basic_plots_recommender(recommender_instance, URM_train, URM_test, output_path_folder, save_on_file,
                            compare_top_pop_points,
                            demographic_list, demographic_list_name, is_compare_top_pop)


def basic_plots_recommender(recommender_instance: BaseRecommender, URM_train, URM_test, output_path_folder,
                            save_on_file, compare_top_pop_points, demographic_list, demographic_list_name,
                            is_compare_top_pop):
    '''
    Plot some graphics concerning the results of a built recommender system
    In particular:
    - Recommendation distribution
    - Comparison against top popular recommender
    - Comparison with user activity
    - Other comparison against an arbitrary number of demographic to be specified

    :param recommender_instance: recommender that will be explored
    :param URM_train: URM train on which to train the recommender
    :param URM_test: URM test on which to test the recommender
    :param save_on_file: if you want to save graphics on file
    :param is_compare_top_pop: if you have to perform the comparison with a top popular recommender
    :param compare_top_pop_points: comparison with the top popular with this points
    :param demographic_list_name: list of addiontal demographics name
    :param demographic_list: list of additional demographics
    :param output_path_folder: where image should be saved, if specified
    :return: None
    '''
    # Plot the trend of the predictions
    plot_recommendation_distribution(recommender_instance, URM_train, at=10)
    fig_rec_distr = plt.gcf()
    fig_rec_distr.show()

    if save_on_file:

        output_path_file = output_path_folder + "recommendation_distribution.png"
        fig_rec_distr.savefig(output_path_file)
        print("Save on file")

    # Plot comparison with top popular
    if is_compare_top_pop:
        top_popular = TopPop(URM_train)
        top_popular.fit()
        plot_comparison_with_top_pop(recommender_instance, top_popular,
                                     compare_top_pop_points, URM_train)
        fig_comparison_with_top_pop = plt.gcf()
        fig_comparison_with_top_pop.show()
        if save_on_file:
            output_path_file = output_path_folder + "comparison_top_pop.png"
            fig_comparison_with_top_pop.savefig(output_path_file)
            print("Save on file")

    # Plot metrics by user popularity
    usr_act = (URM_train > 0).sum(axis=1)
    usr_act = np.squeeze(np.asarray(usr_act))

    res = evaluate_recommender_by_user_demographic(recommender_instance, URM_train, URM_test,
                                                   [10], usr_act, 10)
    plot_metric_results_by_user_demographic(res, metric_name="MAP", cutoff=10,
                                            user_demographic=usr_act,
                                            user_demographic_name="User activity")
    fig_usr_act = plt.gcf()
    fig_usr_act.show()
    if save_on_file:
        output_path_file = output_path_folder + "usr_activity.png"
        fig_usr_act.savefig(output_path_file)
        print("Save on file")

    # Plot other demographics
    if demographic_list is not None:
        for i in range(0, demographic_list):
            res = evaluate_recommender_by_user_demographic(recommender_instance, URM_train, URM_test,
                                                           [10], demographic_list[i], 10)
            plot_metric_results_by_user_demographic(res, metric_name="MAP", cutoff=10,
                                                    user_demographic=demographic_list[i],
                                                    user_demographic_name=demographic_list_name[i])
            fig_dem = plt.gcf()
            fig_dem.show()
            if save_on_file:
                output_path_file = output_path_folder + demographic_list_name[i] + ".png"
                fig_dem.savefig(output_path_file)
                print("Save on file")


def plot_compare_recommenders_user_profile_len(recommender_list,
                                               URM_train, URM_test, recommender_name_list=None, bins=10, metric="MAP",
                                               cutoff=10, save_on_file=False, output_folder_path=""):
    '''
    Plot recommenders in function of the profile lenght of the users.

    Note: recommenders are assumed to be trained on the same URM_train

    :param recommender_list: list of recommenders to be plotted
    :param URM_train: URM on which recommenders have been trained
    :param URM_test: URM on which test the recommenders
    :param recommender_name_list: list of recommenders name. If None, then recommender.RECOMMENDER_NAME will be used
    :param bins: number of bins on which the profile lenght will be discretized
    :param metric: metric considered by the evaluator
    :param cutoff: cutoff on which recommender will be evaluated
    :param save_on_file: if graphics should be saved on file or not
    :param: output_folder_path: destination on which to save the the graphics
    :return: None
    '''
    # Building user profiles groups
    URM_train = sps.csr_matrix(URM_train)
    profile_length = np.ediff1d(URM_train.indptr) # Getting the profile lenght for each user
    sorted_users = np.argsort(profile_length) # Argsorting the user on the basis of their profiles len
    block_size = int(len(profile_length) * (1/bins)) # Calculating the block size, given the desidered number of bins

    group_mean_len = []

    # Print some stats. about the bins
    for group_id in range(0, bins):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, len(profile_length))

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = profile_length[users_in_group]

        group_mean_len.append(int(users_in_group_p_len.mean()))

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))


    plot_compare_recommender_user_group(recommender_list, URM_train, URM_test, block_size, profile_length.size,
                                        sorted_users, profile_length, group_mean_len, recommender_name_list,
                                        bins, metric, cutoff, save_on_file, output_folder_path,
                                        "MAP comparison on profile lens", x_label="User profile mean length")


def plot_compare_recommender_user_group(recommender_list,
                                        URM_train, URM_test,
                                        block_size, total_len, sorted_users, group_metric, group_representative,
                                        recommender_name_list=None, bins=10, metric="MAP",
                                        cutoff=10, save_on_file=False, output_folder_path="",
                                        graph_title="MAP comparison", x_label="User group"):
    '''
    Plot recommenders in function of the profile lenght of the users.

    See the comparison with user profile len for usage.

    Note: recommenders are assumed to be trained on the same URM_train

    :param recommender_list: list of recommenders to be plotted
    :param URM_train: URM on which recommenders have been trained
    :param URM_test: URM on which test the recommenders
    :param block_size: size of the block on which to select the group
    :param total_len: total len of the group metric
    :param sorted_users: users sorted according to the group metric
    :param group_metric: (e.g. profile_lenght)
    :param recommender_name_list: list of recommenders name. If None, then recommender.RECOMMENDER_NAME will be used
    :param bins: number of bins on which the profile lenght will be discretized
    :param group_representative: representative information for each group that will be used in the plot
    :param metric: metric considered by the evaluator
    :param cutoff: cutoff on which recommender will be evaluated
    :param save_on_file: if graphics should be saved on file or not
    :param output_folder_path: destination on which to save the the graphics
    :param graph_title: title of the graph
    :return: None
    '''
    # Initial check on the names of the recommenders
    if recommender_name_list is None:
        recommender_name_list = []
        for recommender in recommender_list:
            if recommender.RECOMMENDER_NAME not in recommender_name_list:
                recommender_name_list.append(recommender.RECOMMENDER_NAME)
            else:
                raise RuntimeError("Recommender names should be unique. Provide a list of names")
    else:
        if len(recommender_name_list) != len(recommender_list):
            raise RuntimeError("Recommender name list and recommender list must have the same size")

    recommender_results = []
    for _ in recommender_list:
        recommender_results.append([])

    # Evaluate the recommenders
    for group_id in range(0, bins):
        start_pos = group_id * block_size
        end_pos = min((group_id + 1) * block_size, total_len)

        users_in_group = sorted_users[start_pos:end_pos]

        users_in_group_p_len = group_metric[users_in_group]

        print("Group {}, average p.len {:.2f}, min {}, max {}".format(group_id,
                                                                      users_in_group_p_len.mean(),
                                                                      users_in_group_p_len.min(),
                                                                      users_in_group_p_len.max()))

        users_not_in_group_flag = np.isin(sorted_users, users_in_group, invert=True)
        users_not_in_group = sorted_users[users_not_in_group_flag]

        evaluator_test = EvaluatorHoldout(URM_test, cutoff_list=[cutoff], ignore_users=users_not_in_group)

        for i, recommender in enumerate(recommender_list):
            results, _ = evaluator_test.evaluateRecommender(recommender)
            recommender_results[i].append(results[cutoff][metric])

    # Plot results
    for i in range(0, len(recommender_list)):
        plt.plot(recommender_results[i], label=recommender_name_list[i])

    plt.title(graph_title)
    plt.xticks(np.arange(bins), group_representative)
    plt.ylabel(metric)
    plt.xlabel(x_label)
    plt.legend()

    if save_on_file:
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        fig = plt.gcf()
        fig.show()
        output_path_file = output_folder_path + "recommender_comparison_" + now + ".png"
        fig.savefig(output_path_file)
        print("Save on file")
