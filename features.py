import numpy as np
import grape
from grape import Graph
from ensmallen import HyperSketchingPy
from sklearn.metrics import accuracy_score, matthews_corrcoef
from cache_decorator import Cache

graph = Graph.from_csv(
    name="LOTUS_with_NCBITaxon",
    node_path="./data/lotus_with_ncbi_nodes.csv",
    node_list_separator="\t",
    node_list_header=True,
    nodes_column_number=0,
    node_list_node_types_column_number=1,
    edge_path="./data/lotus_with_ncbi_edges.csv",
    edge_list_separator="\t",
    edge_list_header=True,
    sources_column_number=0,
    destinations_column_number=1,
    edge_list_edge_types_column_number=2,
    # directed=True,
    directed=False,
)
graph = graph.remove_singleton_nodes()
graph = graph.remove_components(top_k_components=1)

normalize = False
combination = "addition"
number_of_hops = 2

train, test = graph.get_edge_prediction_kfold(
    k_index=0,
    k=5,
    edge_types=["biolink:in_taxon"],
    random_state=42,
)

filtered_train = train.filter_from_names(
    edge_type_names_to_keep=["biolink:in_taxon"],
)

negative = graph.sample_negative_graph(
    source_edge_types_names=["biolink:in_taxon"],
    destination_edge_types_names=["biolink:in_taxon"],
    number_of_negative_samples=graph.get_number_of_edges_from_edge_type_name(
        "biolink:in_taxon"
    )
    * 2,
    sample_edge_types=False,
    only_from_same_component=False,
    use_scale_free_distribution=True,
    random_state=23391 * (10 + 1),
)
neg_train, neg_test = negative.random_holdout(train_size=0.8)


sketching_features = HyperSketchingPy(
    hops=number_of_hops, normalize=normalize, graph=train
)
sketching_features.fit()

# do for positive features
pos_sources = filtered_train.get_directed_source_node_ids()
pos_destinations = filtered_train.get_directed_destination_node_ids()
sk_positive_features = sketching_features.positive(
    sources=pos_sources, destinations=pos_destinations, feature_combination=combination
)

# negative features
neg_sources = neg_train.get_directed_source_node_ids()
neg_destinations = neg_train.get_directed_destination_node_ids()
sk_negative_features = sketching_features.negative(
    sources=neg_sources, destinations=neg_destinations, feature_combination=combination
)

# positive test features
pos_test_srcs = test.get_directed_source_node_ids()
pos_test_dsts = test.get_directed_destination_node_ids()
sk_test_pos_features = sketching_features.unknown(
    sources=pos_test_srcs, destinations=pos_test_dsts, feature_combination=combination
)

# negative test features
neg_test_srcs = neg_test.get_directed_source_node_ids()
neg_test_dsts = neg_test.get_directed_destination_node_ids()
sk_test_neg_features = sketching_features.unknown(
    sources=neg_test_srcs, destinations=neg_test_dsts, feature_combination=combination
)
