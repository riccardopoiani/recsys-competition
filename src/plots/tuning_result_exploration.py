from src.model_management.model_result_reader import *
from src.plots.plot_evaluation_helper import *
from src.model_management.evaluator import *
import numpy as np


def explore_tuning_result(path, recommender_class, URM_train, URM_test,
                          metric='MAP', save_on_file=False, retrain_model=True,
                          is_compare_top_pop = True,
                          compare_top_pop_points=None,
                          demographic_list_name=None, demographic_list=None, output_path_folder=""):
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
        recommender_instance = recommender_class(URM_train)
        recommender_instance.fit(**keywargs)
    else:
        print("Read the best model from file")
        raise NotImplemented("Not implemented feature")

    explore_recommender(recommender_instance, URM_train, URM_test, output_path_folder, save_on_file, compare_top_pop_points,
                        demographic_list, demographic_list_name, is_compare_top_pop)


def explore_recommender(recommender_instance: BaseRecommender, URM_train, URM_test, output_path_folder,
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
