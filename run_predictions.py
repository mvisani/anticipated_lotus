from ensmallen import HyperSketchingPy
from src.models import XGBoost
import pandas as pd
from grape import Graph
from tqdm import tqdm
from src.predict import run_predictions_on_all_molecules_for_one_species

model = XGBoost.load_model("xgboost_model.pkl")


graph = Graph.from_csv(
    name="LOTUS_with_NCBITaxon",
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
    load_edge_list_in_parallel=False,
    load_node_list_in_parallel=False,
)
print("The graph hash is : ", graph.hash())

lotus = pd.read_csv("data/molecules/230106_frozen_metadata.csv.gz", low_memory=False)
lotus["wd_species"] = "wd:" + lotus.organism_wikidata.str.extract(r"(Q\d+)")
lotus["wd_molecule"] = "wd:" + lotus.structure_wikidata.str.extract(r"(Q\d+)")

sketching_features = HyperSketchingPy(
    hops=2,
    normalize=False,
    graph=graph,
)
sketching_features.fit()

for species in tqdm(["wd:Q311176", "wd:Q15550965"]):
    run_predictions_on_all_molecules_for_one_species(
        species=species,
        graph=graph,
        lotus=lotus,
        model=model,
        sketching_features=sketching_features,
    )
