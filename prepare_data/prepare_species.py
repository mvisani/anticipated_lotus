import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

from src.wikidata import taxonomy_in_edges
from tqdm import tqdm
import pandas as pd

tqdm.pandas()


def main():
    # Read the lotus data from the specified file
    lotus = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )

    # Extract the wikidata taxon IDs from the organism_wikidata column
    lotus["wd_taxon"] = lotus["organism_wikidata"].str.extract(r"(Q\d+)")
    lotus["wd_taxon"] = "wd:" + lotus["wd_taxon"]

    # Apply the taxonomy_in_edges function to each unique wd_taxon value. This will get the information of the species on wikidata and return a dataframe with the child and parent columns
    res = (
        lotus.wd_taxon.drop_duplicates()
        .reset_index(drop=True)
        .progress_apply(taxonomy_in_edges)
    )

    # Concatenate the resulting dataframes, drop duplicates and NaN values
    species_phylo = (
        pd.concat(list(res)).drop_duplicates().dropna().reset_index(drop=True)
    )

    # Rename the columns of the species_phylo dataframe
    species_phylo.rename(columns={0: "child", 1: "parent"}, inplace=True)

    # Add "wd:" prefix to the child and parent columns
    species_phylo["child"] = "wd:" + species_phylo["child"].str.extract(r"(Q\d+)")
    species_phylo["parent"] = "wd:" + species_phylo["parent"].str.extract(r"(Q\d+)")

    # Add a edge type column with value "biolink:subclass_of"
    species_phylo["type"] = "biolink:subclass_of"

    # Save the species_phylo dataframe to a CSV file
    species_phylo.to_csv("./data/species/species_edges.csv")

    # Create a dataframe with unique node values and "biolink:OrganismTaxon" type
    species_nodes = pd.DataFrame(
        {
            "node": pd.concat([species_phylo.child, species_phylo.parent])
            .drop_duplicates()
            .values,
            "type": "biolink:OrganismTaxon",
        }
    )

    # Save the species_nodes dataframe to a CSV file
    species_nodes.to_csv("./data/species/species_nodes.csv")


if __name__ == "__main__":
    main()
