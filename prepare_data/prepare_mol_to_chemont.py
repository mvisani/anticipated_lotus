import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

import obonet
import networkx as nx
import pandas as pd


# utils function
def parse_file(file_path):
    id_list = []
    name_list = []

    with open(file_path, "r") as file:
        for line in file:
            if line.startswith("id: "):
                id_list.append(line.strip().split(": ")[1])
            elif line.startswith("name: "):
                name_list.append(line.strip().split(": ")[1])

    df = pd.DataFrame(
        {
            "id": id_list,
            "name": name_list,
        }
    )

    return df


def get_chemont_edges_and_nodes():
    # first read the Classyfire ontology
    chemont = obonet.read_obo("./data/molecules/ChemOnt_2_1.obo")

    # then write it as an edge list
    nx.write_edgelist(
        chemont, "./data/molecules/chemont_edges.csv", data=False, delimiter="\t"
    )

    x = pd.read_csv("./data/molecules/chemont_edges.csv", sep="\t", header=None)
    x[2] = "biolink:subclass_of"
    x.rename(columns={0: "child", 1: "parent", 2: "type"}, inplace=True)
    x.to_csv("./data/molecules/chemont_edges.csv")
    del x
    chemontid_to_name = parse_file("./data/molecules/ChemOnt_2_1.obo")
    chemontid_to_name["type"] = "biolink:ChemicalEntity"
    chemontid_to_name.to_csv("./data/molecules/chemont_nodes.csv", sep="\t")


def main():
    # transform obo data into edge list and node list
    get_chemont_edges_and_nodes()

    # read lotus data
    lotus = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )
    # extract wikidata id
    lotus["wd_structure"] = lotus["structure_wikidata"].str.extract(r"(Q\d+)")
    lotus["wd_structure"] = "wd:" + lotus["wd_structure"]

    # extract direct parent from classyfire
    edges = (
        pd.DataFrame(
            {
                "child": lotus.wd_structure,
                "parent_name": lotus.structure_taxonomy_classyfire_04directparent,
            }
        )
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    # map classyfire parent to chemont id
    chemont_nodes = pd.read_csv(
        "./data/molecules/chemont_nodes.csv", sep="\t", index_col=0
    )
    mapping = {i: j for i, j in zip(chemont_nodes["name"], chemont_nodes["id"])}
    edges["parent"] = edges["parent_name"].map(mapping)
    edges.drop(columns=["parent_name"], inplace=True)
    edges["type"] = "biolink:subclass_of"

    # add chemont nodes
    nodes = (
        pd.DataFrame(
            {
                "node": pd.concat([edges.child, edges.parent]),
                "type": "biolink:ChemicalEntity",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # save edges and nodes
    edges.to_csv("./data/molecules/mol_to_chemont_edges.csv")
    nodes.to_csv("./data/molecules/mol_to_chemont_nodes.csv")


if __name__ == "__main__":
    main()
