import grape
from grape import Graph


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
