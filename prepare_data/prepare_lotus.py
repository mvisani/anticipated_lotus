import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

import pandas as pd


def main():
    lotus = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )

    lotus_agg = (
        lotus.groupby(["structure_wikidata", "organism_wikidata"])
        .size()
        .reset_index(name="count")
    )

    lotus_agg["structure_wikidata"] = "wd:" + lotus_agg[
        "structure_wikidata"
    ].str.extract(r"(Q\d+)")

    lotus_agg["organism_wikidata"] = "wd:" + lotus_agg["organism_wikidata"].str.extract(
        r"(Q\d+)"
    )

    lotus_agg["type"] = "biolink:in_taxon"

    lotus_agg.to_csv("./data/lotus/lotus_edges.csv")

    molecules = (
        pd.DataFrame(
            {
                "node": lotus_agg.structure_wikidata,
                "type": "biolink:ChemicalEntity",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    species = (
        pd.DataFrame(
            {"node": lotus_agg.organism_wikidata, "type": "biolink:OrganismTaxon"}
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    nodes = pd.concat([molecules, species]).reset_index(drop=True)
    nodes.to_csv("./data/lotus/lotus_nodes.csv")


if __name__ == "__main__":
    main()
