import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)


import pandas as pd
from grape import Graph
from grape.datasets.kgobo import NCBITAXON


def main():
    # load lotus
    species = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )
    species["wd_taxon"] = "wd:" + species["organism_wikidata"].str.extract(r"(Q\d+)")

    # keep only the species with a NCBI taxonomy ID
    species = species.dropna(subset="organism_taxonomy_ncbiid").drop_duplicates(
        subset="organism_taxonomy_ncbiid"
    )

    # create a dataframe with the edges
    wd_to_ncbi_edges = pd.DataFrame(
        {
            "wikidata": species["wd_taxon"],
            "ncbi": [
                f"NCBITaxon:{i}"
                for i in species["organism_taxonomy_ncbiid"].astype(int).values
            ],
            "type": "biolink:same_as",
        }
    )

    # create the nodes dataframe
    ncbi_node = pd.concat(
        [
            pd.DataFrame(
                {
                    "node": wd_to_ncbi_edges.wikidata,
                    "type": "biolink:OrganismTaxon",
                }
            ),
            pd.DataFrame(
                {"node": wd_to_ncbi_edges.ncbi, "type": "biolink:OrganismalEntity"}
            ),
        ]
    ).drop_duplicates()

    wd_to_ncbi_graph = Graph.from_pd(
        directed=True,
        edges_df=wd_to_ncbi_edges,
        edge_src_column="wikidata",
        edge_dst_column="ncbi",
        edge_type_column="type",
        nodes_df=ncbi_node,
        node_name_column="node",
        node_type_column="type",
    )

    lotus = Graph.from_csv(
        name="LOTUS",
        node_path="./data/full_lotus_nodes.csv",
        node_list_separator="\t",
        node_list_header=True,
        nodes_column_number=0,
        node_list_node_types_column_number=1,
        edge_path="./data/full_lotus_edges.csv",
        edge_list_separator="\t",
        edge_list_header=True,
        sources_column_number=0,
        destinations_column_number=1,
        edge_list_edge_types_column_number=2,
        # weights_column_number=3,
        directed=True,
    )

    # load the ncbi taxonomy graph
    ncbi_graph = NCBITAXON()

    # merge the lotus graph with the ncbi graph
    lotus_with_ncbi = lotus | wd_to_ncbi_graph | ncbi_graph.to_directed()

    lotus_with_ncbi.dump_nodes(
        path="./data/lotus_with_ncbi_nodes.csv",
        header=True,
        nodes_column_number=0,
        nodes_column="nodes",
        node_types_column_number=1,
        node_type_column="type",
    )

    lotus_with_ncbi.dump_edges(
        path="./data/lotus_with_ncbi_edges.csv",
        header=True,
        directed=True,
        edge_types_column_number=2,
        edge_type_column="edge_type",
    )

    # Remove singleton nodes and components the smaller component
    lotus_with_ncbi_cleaned = lotus_with_ncbi.remove_singleton_nodes()
    lotus_with_ncbi_cleaned = lotus_with_ncbi_cleaned.remove_components(
        top_k_components=1
    )

    lotus_with_ncbi_cleaned.dump_nodes(
        path="./data/lotus_with_ncbi_clean_nodes.csv",
        header=True,
        nodes_column_number=0,
        nodes_column="nodes",
        node_types_column_number=1,
        node_type_column="type",
    )
    lotus_with_ncbi_cleaned.dump_edges(
        path="./data/lotus_with_ncbi_clean_edges.csv",
        header=True,
        directed=True,
        edge_types_column_number=2,
        edge_type_column="edge_type",
    )


if __name__ == "__main__":
    main()
