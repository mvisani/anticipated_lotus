import grape
from grape import Graph
from cache_decorator import Cache
from typing import Type, Dict
from .models.abstract_model import AbstractModel
from .sketching import hyper_sketching_internal
from .experiment import experiments


@Cache(
    cache_dir="experiments/n_hops_{number_of_hops}/combination_{combination}/normalize_{normalize}/ext_{external_holdout_number}/int_{internal_holdout_number}/{_hash}",
    cache_path={
        "params": "{cache_dir}/params.json",
        "performance": "{cache_dir}/performance.json",
    },
)
def internal_holdout(
    graph_without_in_taxon: Graph,
    only_in_taxon_train: Graph,
    number_of_hops: int,
    external_holdout_number: int,
    internal_holdout_number: int,
    combination: str,
    normalize: bool,
    params: dict,
    model_class: Type[AbstractModel],
) -> Dict[str, Dict[str, float]]:
    # sample positive subtraining graph to get a training set and a validation
    (
        only_in_taxon_subtrain,
        only_in_taxon_validation,
    ) = only_in_taxon_train.random_holdout(
        train_size=0.8,
        random_state=(internal_holdout_number + 1) * 314687,
    )
    # create positive sugraph for training again with all nodes and a couple of edges "in_taxon"
    subtrain = only_in_taxon_subtrain | graph_without_in_taxon

    # sample negative subgraph
    # it is possible to sample positive edges in the test set into the negative training. This is by design because we want the model to never know if
    # an edge is positive or not.
    only_in_taxon_negative_subgraph = only_in_taxon_subtrain.sample_negative_graph(
        number_of_negative_samples=only_in_taxon_subtrain.get_number_of_directed_edges(),
        sample_edge_types=False,
        only_from_same_component=False,
        use_scale_free_distribution=True,
        random_state=23391 * (internal_holdout_number + 1),
    )
    (
        only_in_taxon_negative_subtrain,
        only_in_taxon_negative_validation,
    ) = only_in_taxon_negative_subgraph.random_holdout(
        train_size=0.8, random_state=(internal_holdout_number + 1) * 23987
    )

    internal_features = hyper_sketching_internal(
        number_of_hops=number_of_hops,
        external_holdout_number=external_holdout_number,
        internal_holdout_number=internal_holdout_number,
        graph=subtrain,
        graph_with_only_in_taxon_edges=only_in_taxon_subtrain,
        graph_with_only_in_taxon_edges_validation_or_test=only_in_taxon_validation,
        graph_with_only_in_taxon_edges_negative=only_in_taxon_negative_subtrain,
        graph_with_only_in_taxon_edges_validation_or_test_negative=only_in_taxon_negative_validation,
        combination=combination,
        normalize=normalize,
    )

    return {
        "params": params,
        "performance": experiments(
            features=internal_features,
            params=params,
            model_class=model_class,
            random_state=internal_holdout_number,
        ),
    }
