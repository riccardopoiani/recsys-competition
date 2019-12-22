from src.feature.demographics_content import get_profile_demographic
from src.model_management.model_result_reader import *
from src.plots.plot_evaluation_helper import *
from src.model_management.evaluator import *
from course_lib.Base.Evaluation.Evaluator import EvaluatorHoldout
import numpy as np
from datetime import datetime
from src.feature.clustering_utils import cluster_data
from src.utils.general_utility_functions import enable_print


def plot_clustering_demographics(recommender_instance_list: list, URM_train, URM_test, dataframe, metric, cutoff,
                                 recommender_name_list=None,
                                 save_on_file=True,
                                 output_folder_path="",
                                 demographic_name="clustering", n_clusters=10,
                                 n_init=5, init_method="Cao", exclude_cold_users=False):
    """
    Wrapper to plot all the available demographics.

    :param n_init: number of init of the cluster method
    :param init_method: init method for the clustering alg.
    :param n_clusters: number of clasters
    :param recommender_name_list: name of all the recommenders in the list
    :param dataframe: dataframe contained mixed info of UCM + profile lens
    :param cutoff: cutoff at which groups will be evaluated
    :param demographic_name: name of the demographics
    :param recommender_instance_list: list of recommenders
    :param URM_train: URM train on which recommenders are trained
    :param URM_test: URM train on which recommenders should be tested
    :param metric: metric to be considered while evaluating them
    :param save_on_file: if you wish to save on file or not
    :param output_folder_path: output path of the picture. If you wish to save it
    :param exclude_cold_users: if cold users should be excluded from the plot
    :return: None
    """
    # Clustering demographics
    clusters = cluster_data(data=dataframe, n_clusters=n_clusters, n_init=n_init, init_method=init_method)
    cluster_id_list = list(np.arange(len(clusters)))
    demographic_plot(recommender_instance_list, URM_train=URM_train, URM_test=URM_test, cutoff=cutoff,
                     recommender_name_list=recommender_name_list,
                     metric=metric, demographic_name=demographic_name, demographic=clusters, save_on_file=save_on_file,
                     output_folder_path=output_folder_path, demographic_describer_list=cluster_id_list,
                     exclude_cold_users=exclude_cold_users)


def content_plot(recommender_instance_list, URM_train, URM_test, cutoff, metric,
                 content_describer_list: list, content: list, content_name: str, save_on_file=True,
                 output_folder_path="", recommender_name_list=None, exclude_cold_items=False):
    """
    Same thing of demographic plot, but for item contents
    """
    # Initial check on the names of the recommenders
    if recommender_name_list is None:
        recommender_name_list = []
        for recommender in recommender_instance_list:
            if recommender.RECOMMENDER_NAME not in recommender_name_list:
                recommender_name_list.append(recommender.RECOMMENDER_NAME)
            else:
                raise RuntimeError("Recommender names should be unique. Provide a list of names")
    else:
        if len(recommender_name_list) != len(recommender_instance_list):
            raise RuntimeError("Recommender name list and recommender list must have the same size")

    # Recommendation plot
    plt.figure()
    bins = len(content)
    for i, recommender in enumerate(recommender_instance_list):
        results, support = evaluate_recommender_by_item_content(recommender_object=recommender, URM_train=URM_train,
                                                                URM_test=URM_test, cutoff=cutoff, metric=metric,
                                                                content=content, exclude_cold_items=exclude_cold_items)
        plt.plot(results, label=recommender_name_list[i])

    plt.title("Results of {}, accordingly to {}".format(metric, content_name))
    plt.xticks(np.arange(bins), content_describer_list)
    plt.ylabel(metric)
    plt.xlabel(content_name)
    plt.legend()

    # Eventually save on file
    if save_on_file:
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        fig = plt.gcf()
        fig.show()
        if len(recommender_instance_list) == 1:
            output_path_file = output_folder_path + "_" + content_name + "_" + now + ".png"
        else:
            output_path_file = output_folder_path + "_" + content_name + "_comparison_" + now + ".png"
        fig.savefig(output_path_file)
        print("Save on file")

    # Support plot
    plt.figure()
    plt.title("Support picture")
    plt.plot(support)
    plt.xticks(np.arange(bins), content_describer_list)
    plt.ylabel("Number or interactions in the URM_test, escluding those items")
    plt.xlabel("Group id")
    if save_on_file:
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        fig = plt.gcf()
        fig.show()
        if len(recommender_instance_list) == 1:
            output_path_file = output_folder_path + "_" + content_name + "_" + now + ".png"
        else:
            output_path_file = output_folder_path + "_" + content_name + "_comparison_support_" + now + ".png"
        fig.savefig(output_path_file)
        print("Save on file")


def demographic_plot(recommender_instance_list, URM_train, URM_test, cutoff, metric,
                     demographic_describer_list: list, demographic: list, demographic_name: str, save_on_file=True,
                     output_folder_path="", recommender_name_list=None, exclude_cold_users=False):
    """
    Plot the performances of a recommender (or a list of them), on a given metric, based on a certain demographic.

    Note: this function can be used for comparison

    :param recommender_name_list: name of the recommenders
    :param metric: metric to be considered
    :param cutoff: cutoff desired. Only an integer is supported at the moment.
    :param URM_test: URM test on which recommender is evaluated
    :param recommender_instance_list: instance of the recommender to be evaluated
    :param URM_train: URM train on which the recommender is trained
    :param output_folder_path: where to save images, if desired
    :param save_on_file: if you which to save on file, or not
    :param demographic_describer_list: describer of each group of the demographic. This information will be plotted
    on the x-axis of the picture
    :param demographic: list of list. In each list there is
    :param demographic_name: name of the demographic
    :param exclude_cold_users: if cold users should be excluded from the plot
    :return: None
    """
    # Initial check on the names of the recommenders
    if recommender_name_list is None:
        recommender_name_list = []
        for recommender in recommender_instance_list:
            if recommender.RECOMMENDER_NAME not in recommender_name_list:
                recommender_name_list.append(recommender.RECOMMENDER_NAME)
            else:
                raise RuntimeError("Recommender names should be unique. Provide a list of names")
    else:
        if len(recommender_name_list) != len(recommender_instance_list):
            raise RuntimeError("Recommender name list and recommender list must have the same size")

    plt.figure()
    bins = len(demographic)

    for i, recommender in enumerate(recommender_instance_list):
        results, support = evaluate_recommender_by_demographic(recommender_object=recommender, URM_train=URM_train,
                                                               URM_test=URM_test, cutoff=cutoff, metric=metric,
                                                               demographic=demographic,
                                                               exclude_cold_users=exclude_cold_users)
        plt.plot(results, label=recommender_name_list[i])

    plt.title("Results of {}, accordingly to {}".format(metric, demographic_name))
    plt.xticks(np.arange(bins), demographic_describer_list)
    plt.ylabel(metric)
    plt.xlabel(demographic_name)
    plt.legend()

    # Eventually save on file
    if save_on_file:
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        fig = plt.gcf()
        fig.show()
        if len(recommender_instance_list) == 1:
            output_path_file = output_folder_path + "_" + demographic_name + "_" + now + ".png"
        else:
            output_path_file = output_folder_path + "_" + demographic_name + "_comparison_" + now + ".png"
        fig.savefig(output_path_file)
        print("Save on file")

    # Support plot
    plt.figure()
    plt.title("Support picture")
    plt.plot(support)
    plt.xticks(np.arange(bins), demographic_describer_list)
    plt.ylabel("Number or interactions in the URM_test, escluding those items")
    plt.xlabel("Group id")
    if save_on_file:
        now = datetime.now().strftime('%b%d_%H-%M-%S')
        fig = plt.gcf()
        fig.show()
        if len(recommender_instance_list) == 1:
            output_path_file = output_folder_path + "_" + demographic_name + "_" + now + ".png"
        else:
            output_path_file = output_folder_path + "_" + demographic_name + "_comparison_support_" + now + ".png"
        fig.savefig(output_path_file)
        print("Save on file")


def basic_plots_from_tuning_results(path, recommender_class, URM_train, URM_test,
                                    metric='MAP', save_on_file=False, retrain_model=True,
                                    is_compare_top_pop=True,
                                    compare_top_pop_points=None,
                                    demographic_list_name=None, demographic_list=None, output_path_folder="",
                                    ICM=None):
    """
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
    :param demographic_list_name: list of addiontal demographics name @DEPRECATED
    :param demographic_list: list of additional demographics @DEPRECATED
    :param output_path_folder: where image should be saved, if specified
    :return: None
    """
    results = read_folder_metadata(path)

    print("hello")

    # Create folder if it does not exist
    if save_on_file:
        try:
            if not os.path.exists(output_path_folder):
                os.mkdir(output_path_folder)
        except FileNotFoundError as e:
            os.makedirs(output_path_folder)

    # PLOT WHERE SAMPLES HAVE BEEN TAKEN
    for count, r in enumerate(results):
        arhr_index = -1
        for j in range(0, len(r.columns)):
            if r.columns[j] == 'ARHR':
                arhr_index = j
                break
        dimension_list = r.columns[0:arhr_index]

        # Plot
        if len(dimension_list) > 1:
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
        print("Read the best model from file...")
        raise NotImplemented("Not implemented feature")

    # Store on file keywargs of the recommender under consideration
    if save_on_file:
        output_path_file = output_path_folder + "keywargs.txt"
        f = open(output_path_file, "w")
        f.write("Recommender name: " + recommender_instance.RECOMMENDER_NAME + "\n")
        f.write(str(keywargs))
        f.write("\n")
        f.close()

    basic_plots_recommender(recommender_instance, URM_train, URM_test, output_path_folder, save_on_file,
                            compare_top_pop_points,
                            demographic_list, demographic_list_name, is_compare_top_pop)


def basic_plots_recommender(recommender_instance: BaseRecommender, URM_train, URM_test, output_path_folder,
                            save_on_file, compare_top_pop_points, demographic_list, demographic_list_name,
                            is_compare_top_pop=True, is_plot_hexbin=False):
    """
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
    :param is_plot_hexbin: if True, it plots the hexbin of the recommendations
    :param compare_top_pop_points: comparison with the top popular with this points
    :param demographic_list_name: list of additional demographics name @DEPRECATED
    :param demographic_list: list of additional demographics @DEPRECATED
    :param output_path_folder: where image should be saved, if specified
    :return: None
    """
    # Create directories if saving files
    if save_on_file:
        try:
            if not os.path.exists(output_path_folder):
                os.mkdir(output_path_folder)
        except FileNotFoundError as e:
            os.makedirs(output_path_folder)

    # Plot the trend of the predictions
    plot_recommendation_distribution(recommender_instance, URM_train, cutoff=10)
    fig_rec_distr = plt.gcf()
    fig_rec_distr.show()

    if save_on_file:
        file_name = "recommendation_distribution.png"
        output_path_file = output_path_folder + file_name
        fig_rec_distr.savefig(output_path_file)
        print("Plot recommendations distribution: saved as {}".format(file_name))

    # Plot comparison with top popular
    if is_compare_top_pop:
        if compare_top_pop_points is None:
            compare_top_pop_points = [10, 100, 500, 1000]

        block_print()  # Disable useless prints
        top_popular = TopPop(URM_train)
        top_popular.fit()
        enable_print()

        plot_comparison_with_top_pop(recommender_instance, top_popular, compare_top_pop_points, URM_train)
        fig_comparison_with_top_pop = plt.gcf()
        fig_comparison_with_top_pop.show()
        if save_on_file:
            file_name = "comparison_top_pop.png"
            output_path_file = output_path_folder + file_name
            fig_comparison_with_top_pop.savefig(output_path_file)
            print("Plot comparison with TopPop: saved as {}".format(file_name))

    # Plot metrics by user popularity
    user_activity = (URM_train > 0).sum(axis=1)
    user_activity = np.squeeze(np.asarray(user_activity))

    print("Plot metric results by user activity: init")
    res = evaluate_recommender_by_user_demographic(recommender_instance, URM_train, URM_test, [10], user_activity, 10)
    plot_metric_results_by_user_demographic(res, metric_name="MAP", cutoff=10,
                                            user_demographic=user_activity,
                                            user_demographic_name="User activity")
    fig_usr_act = plt.gcf()
    fig_usr_act.show()
    if save_on_file:
        file_name = "user_activity.png"
        output_path_file = output_path_folder + file_name
        fig_usr_act.savefig(output_path_file)
        print("Plot metric by user activity: saved as {}".format(file_name))

    # Plot other demographics
    if demographic_list is not None:
        for i in range(0, len(demographic_list)):
            print("Plot metric results by demographic: {}".format(demographic_list_name[i]))
            res = evaluate_recommender_by_user_demographic(recommender_instance, URM_train, URM_test,
                                                           [10], demographic_list[i], 10)
            plot_metric_results_by_user_demographic(res, metric_name="MAP", cutoff=10,
                                                    user_demographic=demographic_list[i],
                                                    user_demographic_name=demographic_list_name[i])
            fig_dem = plt.gcf()
            fig_dem.show()
            if save_on_file:
                file_name = demographic_list_name[i] + ".png"
                output_path_file = output_path_folder + file_name
                fig_dem.savefig(output_path_file)
                print("Plot metric by demographic: saved as {}".format(file_name))

    # Plot hexbin recommendations plot
    if is_plot_hexbin:
        plot_hexbin(recommender_instance, URM_test, 10)
        fig_usr_act = plt.gcf()
        fig_usr_act.show()
        if save_on_file:
            file_name = "hexbin_recommendations.png"
            output_path_file = output_path_folder + file_name
            fig_usr_act.savefig(output_path_file)
            print("Plot hexbin recommendations: saved as {}".format(file_name))


def plot_hexbin(recommender_object, URM_test, cutoff, batches=20):
    x_all = []
    y_all = []
    users_batches_list = np.array_split(np.arange(URM_test.shape[0]), batches)
    with tqdm(desc="Plot hexbin recommendations", total=URM_test.shape[0]) as p_bar:
        for user_batch in users_batches_list:
            recommendations = recommender_object.recommend(user_batch, cutoff=cutoff)
            x = np.repeat(user_batch, cutoff)
            y = np.array(recommendations).flatten()
            x_all = np.concatenate([x_all, x])
            y_all = np.concatenate([y_all, y])

            p_bar.update(len(user_batch))
    fig, ax = plt.subplots()
    hb = ax.hexbin(x_all, y_all, gridsize=30, mincnt=10)
    ax.set_title("Hexbin recommendations")
    ax.set_xlabel("User index")
    ax.set_ylabel("Item index")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label('counts')
    fig_usr_act = plt.gcf()
    fig_usr_act.show()

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
    block_size, profile_length, sorted_users, group_mean_len = get_profile_demographic(URM_train, bins)

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
                print(recommender_name_list)
                print(recommender.RECOMMENDER_NAME)
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
