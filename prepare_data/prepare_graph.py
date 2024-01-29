import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)


from grape import Graph


def main():
    lotus = Graph.from_csv(
        node_path="./data/lotus/lotus_nodes.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/lotus/lotus_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        # weights_column_number=3,
        edge_list_edge_types_column_number=4,
        # directed=False,
        directed=True,
    )

    species = Graph.from_csv(
        node_path="./data/species/species_nodes.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/species/species_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # weights_column_number=4,
        directed=True,
        # directed=False,
    )

    molecules_to_chemont = Graph.from_csv(
        node_path="./data/molecules/mol_to_chemont_nodes.csv",
        node_list_separator=",",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/molecules/mol_to_chemont_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # weights_column_number=4,
        directed=True,
        # directed=False,
    )

    chemont = Graph.from_csv(
        node_path="./data/molecules/chemont_nodes.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=3,
        edge_path="./data/molecules/chemont_edges.csv",
        edge_list_separator=",",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # weights_column_number=4,
        # directed=False,
        directed=True,
    )

    molecules_to_np = Graph.from_csv(
        node_path="./data/molecules/mol_to_np_nodes.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=1,
        node_list_node_types_column_number=2,
        edge_path="./data/molecules/mol_to_np_edges.csv",
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column_number=1,
        destinations_column_number=2,
        edge_list_edge_types_column_number=3,
        # directed=False,
        directed=True,
    )

    chemicals = chemont | molecules_to_chemont

    chemical_with_np_classifier = chemicals | molecules_to_np

    chemicals_and_lotus = chemical_with_np_classifier | lotus

    full_graph = chemicals_and_lotus | species

    full_graph.dump_nodes(
        path="./data/full_lotus_nodes.csv",
        header=True,
        nodes_column_number=0,
        nodes_column="nodes",
        node_types_column_number=1,
        node_type_column="type",
    )

    full_graph.dump_edges(
        path="./data/full_lotus_edges.csv",
        header=True,
        directed=True,
        edge_types_column_number=2,
        edge_type_column="type",
    )


if __name__ == "__main__":
    main()
