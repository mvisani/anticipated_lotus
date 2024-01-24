import grape
from cache_decorator import Cache


def hyper_sketching(
    graph,
    graph_with_only_in_taxon_edges,
    graph_with_only_in_taxon_edges_validation_or_test,
    graph_with_only_in_taxon_edges_negative,
    graph_with_only_in_taxon_edges_validation_or_test_negative,
    number_of_hops,
    combination="addition",
    normalize=False,
):
    from ensmallen import HyperSketchingPy

    sketching_features = HyperSketchingPy(
        hops=number_of_hops,
        normalize=normalize,
        graph=graph,
    ).fit()

    # sketching for positive training edges
    pos_sources = graph_with_only_in_taxon_edges.get_directed_source_node_ids()
    pos_destinations = (
        graph_with_only_in_taxon_edges.get_directed_destination_node_ids()
    )
    sk_positive_features = sketching_features.positive(
        sources=pos_sources,
        destinations=pos_destinations,
        feature_combination=combination,
    )

    # sketching for positive validation or testing
    pos_test_srcs = (
        graph_with_only_in_taxon_edges_validation_or_test.get_directed_source_node_ids()
    )
    pos_test_dsts = (
        graph_with_only_in_taxon_edges_validation_or_test.get_directed_destination_node_ids()
    )
    sk_test_pos_features = sketching_features.unknown(
        sources=pos_test_srcs,
        destinations=pos_test_dsts,
        feature_combination=combination,
    )

    # sketching for training negatives
    neg_sources = graph_with_only_in_taxon_edges_negative.get_directed_source_node_ids()
    neg_destinations = (
        graph_with_only_in_taxon_edges_negative.get_directed_destination_node_ids()
    )
    sk_negative_features = sketching_features.negative(
        sources=neg_sources,
        destinations=neg_destinations,
        feature_combination=combination,
    )

    # sketching for negative test or validation
    neg_test_srcs = (
        graph_with_only_in_taxon_edges_validation_or_test_negative.get_directed_source_node_ids()
    )
    neg_test_dsts = (
        graph_with_only_in_taxon_edges_validation_or_test_negative.get_directed_destination_node_ids()
    )
    sk_test_neg_features = sketching_features.unknown(
        sources=neg_test_srcs,
        destinations=neg_test_dsts,
        feature_combination=combination,
    )
    return {
        "train_positive": sk_positive_features,
        "test_positive": sk_test_pos_features,
        "train_negative": sk_negative_features,
        "test_negative": sk_test_neg_features,
    }


@Cache(
    cache_dir="experiments/n_hops_{number_of_hops}/combination_{combination}/normalize_{normalize}/ext_{external_holdout_number}/{_hash}",
    cache_path={
        "train_positive": "{cache_dir}/train_positive.npz",
        "test_positive": "{cache_dir}/test_positive.npz",
        "train_negative": "{cache_dir}/train_negative.npz",
        "test_negative": "{cache_dir}/test_negative.npz",
    },
)
def hyper_sketching_outer(
    number_of_hops,
    external_holdout_number,
    graph,
    graph_with_only_in_taxon_edges,
    graph_with_only_in_taxon_edges_validation_or_test,
    graph_with_only_in_taxon_edges_negative,
    graph_with_only_in_taxon_edges_validation_or_test_negative,
    combination="addition",
    normalize=False,
):
    return hyper_sketching(
        graph,
        graph_with_only_in_taxon_edges,
        graph_with_only_in_taxon_edges_validation_or_test,
        graph_with_only_in_taxon_edges_negative,
        graph_with_only_in_taxon_edges_validation_or_test_negative,
        number_of_hops,
        combination=combination,
        normalize=normalize,
    )


@Cache(
    cache_dir="experiments/n_hops_{number_of_hops}/combination_{combination}/normalize_{normalize}/ext_{external_holdout_number}/int_{internal_holdout_number}/{_hash}",
    cache_path={
        "train_positive": "{cache_dir}/subtrain_positive.npz",
        "test_positive": "{cache_dir}/validation_positive.npz",
        "train_negative": "{cache_dir}/subtrain_negative.npz",
        "test_negative": "{cache_dir}/validation_negative.npz",
    },
)
def hyper_sketching_internal(
    number_of_hops,
    external_holdout_number,
    internal_holdout_number,
    graph,
    graph_with_only_in_taxon_edges,
    graph_with_only_in_taxon_edges_validation_or_test,
    graph_with_only_in_taxon_edges_negative,
    graph_with_only_in_taxon_edges_validation_or_test_negative,
    combination="addition",
    normalize=False,
):
    return hyper_sketching(
        graph,
        graph_with_only_in_taxon_edges,
        graph_with_only_in_taxon_edges_validation_or_test,
        graph_with_only_in_taxon_edges_negative,
        graph_with_only_in_taxon_edges_validation_or_test_negative,
        number_of_hops,
        combination=combination,
        normalize=normalize,
    )
