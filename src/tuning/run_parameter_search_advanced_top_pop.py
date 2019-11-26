from skopt.space import Integer, Categorical, Real


def run_parameter_search_advanced_top_pop(verbose=0, seed=69420):

    hyperparameters_range_dictionary = {"clustering_method": Categorical(['kmodes', 'kproto']),
                                        'n_clusters': Integer(1, 20),
                                        'init_method': Categorical(["Huang", "random", "Cao"]), 'verbose': 0,
                                        'seed': seed}

    # TODO
    # Run the search, considering only cold users for the validation



