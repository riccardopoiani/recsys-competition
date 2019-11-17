import numpy as np
import pandas as pd
import scipy.sparse as sps
from matplotlib import pyplot as plt
from course_lib.Base.BaseRecommender import BaseRecommender
from course_lib.Base.NonPersonalizedRecommender import TopPop
from src.model_management.model_comparison import compare_with_top_popular_recommender


def plot_metric_results_by_user_demographic(results: list, metric_name: str = "MAP", cutoff: int = 10,
                                            user_demographic: np.ndarray = None, user_demographic_name = "user demographic"):
    """
    Plot barplot of one metric from the dict of metric results generated user_demographic evaluate_recommender_by_user_demographic
    and the actual user demographic if the array "user_demographic" is passed

    :param results: list of dicts of metric results generated user_demographic evaluate_recommender_by_user_demographic
    :param metric_name: the name of the metric to plot
    :param cutoff: the cutoff used during evaluate_recommender_by_user_demographic
    :param user_demographic: if it is not "None", then the figure will also plot the user demographic
    :param user_demographic_name: name of the user demographic passed
    :return: None
    """
    metrics = _get_metric_result(results, metric_name, cutoff)
    bins = len(metrics)

    max_id = len(user_demographic) if user_demographic is not None else 100

    plt.figure()
    plt.bar([i for i in range(0, int(np.ceil(max_id / bins)) * bins, int(np.ceil(max_id / bins)))],
            metrics, width=np.ceil(max_id / bins), align='edge')
    plt.suptitle("Barplot of {} for different subset of user based on {}".format(metric_name, user_demographic_name))
    plt.title("Overall {} {:.6f}".format(metric_name, np.array(metrics).mean()))
    plt.xlabel("User Index")
    plt.ylabel(metric_name)

    if user_demographic is not None:
        user_demographic = np.sort(user_demographic)
        ax2 = plt.twinx()
        ax2.plot(user_demographic, 'r.', label='{}'.format(user_demographic_name.capitalize()))
        ax2.set_ylabel("Number of user interactions")
        ax2.legend()

    plt.show()


def _get_metric_result(results: list, metric_name: str, cutoff: int):
    if cutoff not in results[0].keys():
        raise KeyError("The cutoff does not exist, please use the one used to evaluate")
    if metric_name not in results[0][cutoff].keys():
        raise KeyError("The metric_name does not exist, the available ones are: {}".format(results[0][cutoff].keys()))

    return [result[cutoff][metric_name] for result in results]


def plot_sample_evaluations(result: pd.DataFrame, dimensions_list: list, metric_name="MAP", maximize=True, bins=20):
    """
    (Copy of skopt.plots.plot_evaluations)
    Visualize the order in which points where sampled.

    The scatter plot matrix shows cutoff which points in the search
    space and in which order samples were evaluated. Pairwise
    scatter plots are shown on the off-diagonal for each
    dimension of the search space. The order in which samples
    were evaluated is encoded in each point's color.
    The diagonal shows a histogram of sampled values for each
    dimension. A red point indicates the found minimum.

    Note: search spaces that contain `Categorical` dimensions are
          currently not supported by this function.

    Note: This function is dependent from read_tuning_metadata_file!

    :param result: [pd.DataFrame] the result inside a pd.DataFrame for which to create the scatter plot matrix
    :param bins: [int, bins=20] number of bins to use for histograms on the diagonal
    :param metric_name: [str] the name of the metric to plot
    :param dimensions: [list of str] labels of the dimension variables
    :return: the matplotlib axes
    """

    dimensions = len(dimensions_list)
    best_idx = result[metric_name].idxmax() if maximize else result[metric_name].idxmin()
    best_parameters = result.iloc[best_idx]
    order = range(result.shape[0])
    fig, ax = plt.subplots(dimensions, dimensions,
                           figsize=(2 * dimensions, 2 * dimensions))

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.1, wspace=0.1)

    for i in range(dimensions):
        for j in range(dimensions):
            dim_i = dimensions_list[i]
            dim_j = dimensions_list[j]
            if i == j:
                if result[dim_i].dtype == "bool":
                    counts = result[dim_i].value_counts()
                    ax[i, i].bar(counts.index.astype('str'), counts.values)

                else:
                    ax[i, i].hist(result[dim_i], bins=bins,
                                  range=(result[dim_i].min(), result[dim_i].max()))
                ax[i, i].set_xlabel(dim_i)
                ax[i, i].set_ylabel("Number of samples")
                ax[i, i].xaxis.set_ticks_position("top")
                ax[i, i].yaxis.set_ticks_position("right")
                ax[i, i].xaxis.set_label_position("top")
                ax[i, i].yaxis.set_label_position("right")

            # lower triangle
            elif i > j:
                ax[i, j].scatter(result[dim_j], result[dim_i], c=order,
                                 s=40, lw=0., cmap='viridis')
                ax[i, j].scatter(best_parameters[dim_j], best_parameters[dim_i],
                                 c=['r'], s=20, lw=0.)
                if i == dimensions - 1:
                    ax[i, j].set_xlabel(dim_j)
                    if j != 0:
                        ax[i, j].yaxis.set_visible(False)

                if j == 0:
                    ax[i, j].set_ylabel(dim_i)
                    if i != dimensions - 1:
                        ax[i, j].xaxis.set_visible(False)

                # hide upper triangle
                ax[j, i].set_visible(False)
    return fig, ax


def plot_recommendation_distribution(recommender_object: BaseRecommender, URM, at=5,
                                     graph_title="Number of recommendations", x_label="Items"):
    '''
    Show the distributions of the recommendations made by a certain recommender object on the given URM.
    In the notebooks of the lecture, the URM used is URM_all.

    :param x_label: label of the x axis of the graph
    :param graph_title: title of the graph
    :param recommender_object: recommender that will make the recommendations
    :param URM: URM on which the recommender will do the recommendations
    :return: graph of the recommendation distributions
    '''

    x_tick = np.arange(URM.shape[1])
    counter = np.zeros(URM.shape[1])
    for user_id in range(URM.shape[0]):
        recs = recommender_object.recommend(user_id, cutoff=at)
        counter[recs] += 1
        if user_id % 10000 == 0:
            print("Recommended to user {}/{}".format(user_id, URM.shape[0]))

    plt.plot(x_tick, np.sort(counter)[::-1])
    plt.ylabel(graph_title)
    plt.xlabel(x_label)
    plt.show()


def plot_comparison_with_top_pop(recommender_object: BaseRecommender, top_popular: TopPop, list_of_num_max_points: list,
                                 URM, cutoff=10, graph_title="Comparison with Top Popular recommender"):
    '''
    Plot the comparison with the top popular recommender and compute the comparison for a number of recommendations
    specified in list_of_num_max_points


    :param recommender_object: recommender to compare
    :param top_popular: top popular recommender
    :param list_of_num_max_points: contains the number of elements that the top pop will recommend (i.e. its cutoff)
    :param graph_title: Title of the graph
    :return: y values of the graph
    '''
    list_of_num_max_points.sort()
    y_ticks = []
    for item in list_of_num_max_points:
        comparison_results = compare_with_top_popular_recommender(top_pop_recommender=top_popular, recommender_to_compare=recommender_object
                                             , cutoff=cutoff, num_most_pop=item, URM=URM)
        y_ticks.append(comparison_results)

    plt.plot(list_of_num_max_points, y_ticks)
    plt.ylabel('Average number of items in common with top-popular')
    plt.xlabel('Number of items recommended by top-popular')
    plt.title(graph_title)
    plt.show()

    return y_ticks


def plot_ordered_metrics(metrics_df : pd.DataFrame):
    """
    Plot ordered metrics of the dataframe "metrics_df" containing all the metrics about single user obtained by
    src.model_management.evaluator.get_singular_user_metrics

    :param metrics_df: pandas.DataFrame containing all the metrics about single user
    :return: None
    """

    plt.plot(metrics_df.sort_values("AP", ascending=False)["AP"].values, label="AP")
    plt.plot(metrics_df.sort_values("precision", ascending=False)["precision"].values, label="precision")
    plt.plot(metrics_df.sort_values("recall", ascending=False)["recall"].values, label="recall")
    plt.title("Metric of users with at least one element in URM test")
    plt.xlabel("User Index")
    plt.ylabel("Metric value")
    plt.legend()
    plt.show()


def plot_popularity_discretized(array, threshold_list, y_label="Percentage of element popularity"):
    '''
    Plot popularity of a certain array discretized.

    It is recommended to call with function with threshold_list[0] = 0.
    Morevoer, the threshold_list is assumed to be ordered


    :param array: array of popularity (item or users), that contains, in each position how many interactions
    a certain user or a certain items have
    :param threshold_list: list of threshold that will be used for the plot. In particular, it will be
    plotted popularity ranged in each interval of threshold_list[i-1] and threshold_list[i]
    :param y_label: y label of the graph
    :return: array containing information of the plot: the first one has asbolute values, the second one
    a normalized version
    '''
    condition_list = []

    t0 = threshold_list[0]
    cond = array == threshold_list[0]
    condition_list.append(cond)

    t_prev = t0
    for i in range(1, len(threshold_list)):
        cond = np.logical_and((array > t_prev), (array <= threshold_list[i]))
        condition_list.append(cond)
        t_prev = threshold_list[i]

    shape_list = []
    for c in condition_list:
        temp = np.extract(c, array)
        shape_list.append(temp.shape[0])

    print("We have " + str(shape_list[0]) + " array elems with a " + str(threshold_list[0]) + " interactions")
    for i in range (1, len(shape_list)):
        print("We have " + str(shape_list[i]) + " array elems with interactions in (" + str(threshold_list[i-1]) + ", " +str(threshold_list[i]) + "]")

    shape_arr = np.array(shape_list)
    total = shape_arr.sum()
    normalized_shape = np.divide(shape_arr, total)
    t_str = []
    for t in threshold_list:
        t_str.append(str(t))

    plt.bar(t_str, normalized_shape, align='center', alpha=0.5)

    #plt.xticks(y_pos, objects)
    plt.ylabel(y_label)
    plt.xlabel("Threshold")
    plt.title('Discretized graph')

    plt.show()

    return shape_arr, normalized_shape