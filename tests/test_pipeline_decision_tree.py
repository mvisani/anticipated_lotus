import grape
from grape import Graph
from src import run
from src.models import DecisionTree


def test_pipeline():
    graph = Graph.from_csv(
        name="test_graph",
        node_path="./tests/node_list.csv",
        node_list_header=False,
        nodes_column_number=0,
        edge_path="./tests/edge_list.csv",
        edge_list_separator=",",
        sources_column_number=0,
        destinations_column_number=1,
        edge_list_edge_types_column_number=2,
        directed=False,
    )

    result = run(
        graph=graph,
        number_of_external_holdouts=2,
        number_of_internal_holdouts=2,
        number_of_hops=1,
        combination="addition",
        normalize=False,
        model_class=DecisionTree,
        max_evals=1000,
    )

    # assert result.shape == (1, 26)