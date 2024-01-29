import os
import sys

# Get the absolute path of the current script
current_script_path = os.path.abspath(__file__)

# Navigate to the project root directory
project_root = os.path.dirname(os.path.dirname(current_script_path))
sys.path.insert(0, project_root)

import json
import pandas as pd


def main():
    # open the json file with the NPClassifier index
    with open("./data/molecules/NPClassifier_index.json", "r") as total:
        index = json.load(total)

    # create a dataframe with the class numbers and class names
    class_ = pd.DataFrame(index["Class"], index=["class_num"]).T
    class_["class"] = class_.index
    class_.reset_index(inplace=True, drop=True)

    # create a dataframe with the superclass numbers and superclass names
    superclass = pd.DataFrame(index["Superclass"], index=["superclass_num"]).T
    superclass["superclass"] = superclass.index

    # create a dataframe with the pathway numbers and pathway names
    pathway = pd.DataFrame(index["Pathway"], index=["pathway_num"]).T
    pathway["pathway"] = pathway.index

    # create a dataframe with the class hierarchy
    class_hierarchy = pd.DataFrame(index["Class_hierarchy"]).T
    class_hierarchy = class_hierarchy.applymap(lambda x: x[0])
    class_hierarchy["class_num"] = class_hierarchy.index
    class_hierarchy.class_num = class_hierarchy.class_num.astype(int)

    # merge the dataframes to create a dataframe with the class hierarchy
    step1 = pd.merge(class_, class_hierarchy, on="class_num")
    step2 = pd.merge(step1, superclass, left_on="Superclass", right_on="superclass_num")
    df = pd.merge(step2, pathway, left_on="Pathway", right_on="pathway_num")

    # drop the columns that are not needed
    df = df.drop(
        columns=["class_num", "Pathway", "Superclass", "superclass_num", "pathway_num"]
    )

    # create a dataframe with the edges
    np_classifier_edges = (
        pd.DataFrame(
            {
                "child": pd.concat([df["class"], df["superclass"]]),
                "parent": pd.concat([df["superclass"], df["pathway"]]),
                "type": "biolink:subclass_of",
            }
        )
        .drop_duplicates()
        .reset_index(drop=True)
    )

    # read the lotus data from the specified file
    lotus = pd.read_csv(
        "./data/molecules/230106_frozen_metadata.csv.gz", low_memory=False
    )
    lotus["wd"] = "wd:" + lotus["structure_wikidata"].str.extract(r"(Q\d+)")

    # create a dataframe with the wikidata taxon IDs and the NPClassifier class names this makes the edges list
    mol_to_np = (
        lotus[["wd", "structure_taxonomy_npclassifier_03class"]]
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    # add a column with the edge type
    mol_to_np["type"] = "biolink:subclass_of"

    mol_to_np.rename(
        columns={"wd": "child", "structure_taxonomy_npclassifier_03class": "parent"},
        inplace=True,
    )

    # concatenate the two dataframes and drop duplicates and NaN values
    mol_to_np_edges = (
        pd.concat([mol_to_np, np_classifier_edges])
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    mol_to_np_nodes = (
        pd.DataFrame(
            {
                "node": pd.concat([mol_to_np_edges.child, mol_to_np_edges.parent]),
                "type": "biolink:ChemicalEntity",
            }
        )
        .drop_duplicates()
        .dropna()
        .reset_index(drop=True)
    )

    mol_to_np_edges.to_csv("./data/molecules/mol_to_np_edges.csv", sep="\t")
    mol_to_np_nodes.to_csv("./data/molecules/mol_to_np_nodes.csv", sep="\t")


if __name__ == "__main__":
    main()
