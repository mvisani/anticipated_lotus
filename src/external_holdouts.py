import grape
from grape import Graph
from cache_decorator import Cache
from typing import Type, Dict
from .models.abstract_model import AbstractModel
from .sketching import hyper_sketching_outer
from .experiment import experiments
import numpy as np
from .internal_holdouts import internal_holdout
from .bayesian_optimization import hyperopt_optimization


@Cache(
    cache_dir="experiments/n_hops_{number_of_hops}/combination_{combination}/normalize_{normalize}/ext_{external_holdout_number}/{_hash}",
    cache_path={
        "optimal_params": "{cache_dir}/optimal_params.json",
        "performance": "{cache_dir}/performance.json",
    },
)
def external_holdout(
    graph_without_in_taxon: Graph,
    graph_with_only_in_taxon: Graph,
    number_of_hops: int,
    external_holdout_number: int,
    number_of_internal_holdouts: int,
    combination: str,
    normalize: bool,
    model_class: Type[AbstractModel],
):
    # split the graph containg only "in_taxon" edges into training and testing
    # this will be tjhe positive training and testing
    only_in_taxon_train, only_in_taxon_test = graph_with_only_in_taxon.random_holdout(
        train_size=0.8,
        random_state=(external_holdout_number + 1) * 5741775,
    )

    # for the training we want the entire graph with roughly 80% of edges "in_taxon"
    train = only_in_taxon_train | graph_without_in_taxon

    # sample negative edges with the assumption that the majority of edges will be negative.
    # again we split into training and testing
    only_in_taxon_negative = graph_with_only_in_taxon.sample_negative_graph(
        number_of_negative_samples=graph_with_only_in_taxon.get_number_of_directed_edges(),
        sample_edge_types=False,
        only_from_same_component=False,
        use_scale_free_distribution=True,
        random_state=23391 * (external_holdout_number + 1),
    )

    # split negatives into training and test
    (
        only_in_taxon_negative_train,
        only_in_taxon_negative_test,
    ) = only_in_taxon_negative.random_holdout(
        train_size=0.8, random_state=(external_holdout_number + 1) * 23987
    )

    external_features = hyper_sketching_outer(
        number_of_hops=number_of_hops,
        external_holdout_number=external_holdout_number,
        graph=train,
        graph_with_only_in_taxon_edges=only_in_taxon_train,
        graph_with_only_in_taxon_edges_validation_or_test=only_in_taxon_test,
        graph_with_only_in_taxon_edges_negative=only_in_taxon_negative_train,
        graph_with_only_in_taxon_edges_validation_or_test_negative=only_in_taxon_negative_test,
        combination=combination,
        normalize=normalize,
    )

    def objective(params):
        performance = []
        for inner_holdout_number in range(number_of_internal_holdouts):
            # needs to integrate the internal holdout function
            # the internal holdout function returns a dictionary with the performance of the model
            # we want the to return the mcc of the m
            performance.append(
                internal_holdout(
                    graph_without_in_taxon,
                    only_in_taxon_train,
                    number_of_hops,
                    external_holdout_number,
                    inner_holdout_number,
                    combination,
                    normalize,
                    params,
                    model_class,
                )
            )

        # we do minus the mean because hyperopt only has a fmin function that minimizes the space
        return -np.mean([p["test"]["mcc"] for p in performance])

    # then the bayesian optimization needs to take as input the objective function and the hyperparameter space
    # and return the optimal hyperparameters

    ## hyperopt optimization
    optimal_params = hyperopt_optimization(
        objective=objective, space=model_class.search_space()
    )

    return {
        "optimal_params": optimal_params,
        "performance": experiments(
            features=external_features,
            params=optimal_params,
            model_class=model_class,
            random_state=external_holdout_number,
        ),
    }
