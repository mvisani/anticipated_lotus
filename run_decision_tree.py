from grape import Graph
from src import run
from src.models import DecisionTree

# TODO change graph to a dumped version of graph.remove_singleton_nodes()


def main():
    graph = Graph.from_csv(
        name="lotus_with_ncbi",
        node_path="./data/lotus_with_ncbi_clean_nodes.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=0,
        node_list_node_types_column_number=1,
        edge_path="./data/lotus_with_ncbi_clean_edges.csv",
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column_number=0,
        destinations_column_number=1,
        edge_list_edge_types_column_number=2,
        # directed=True,
        directed=False,
    )
    run(
        graph=graph,
        number_of_external_holdouts=10,
        number_of_internal_holdouts=3,
        number_of_hops=2,
        combination="addition",
        normalize=False,
        model_class=DecisionTree,
        max_evals=2,
    )


if __name__ == "__main__":
    main()
