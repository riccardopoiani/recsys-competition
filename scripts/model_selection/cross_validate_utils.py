def get_seed_list():
    return [6910, 1996, 2019, 153, 12, 5, 1010, 9999, 666, 467]


def write_results_on_file(file_path, recommender_name, recommender_fit_parameters, num_folds, seed_list, results):
    with open(file_path, "w") as f:
        f.write("Recommender class: {}\n".format(recommender_name))
        f.write("Recommender fit parameters: {}\n".format(recommender_fit_parameters))
        f.write("Number of folds: {}\n".format(num_folds))
        f.write("Seed list: {}\n\n".format(str(seed_list)))
        f.write(str(results))
