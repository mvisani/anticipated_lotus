from src.models import XGBoost
import pandas as pd
import numpy as np
from grape import Graph
from cache_decorator import Cache
from typing import Type
from .models.abstract_model import AbstractModel


@Cache(
    cache_dir="predictions/{model}/{species}/",
    cache_path="{cache_dir}/{_hash}.csv.gz",
    args_to_ignore=["sketching_features"],
    use_approximated_hash=True,
)
def run_predictions_on_all_molecules_for_one_species(
    species: str,
    graph: Graph,
    lotus: pd.DataFrame,
    model: Type[AbstractModel],
    sketching_features,
) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "molecule_wikidata": [i for i in sorted(lotus.wd_molecule.unique())],
        }
    )
    df["species_wikidata"] = species
    df["species_name"] = lotus[lotus.wd_species == species].organism_name.unique()[0]

    molecules_id = graph.get_node_ids_from_node_names(df.molecule_wikidata)
    species_id = graph.get_node_ids_from_node_names(df.species_wikidata)

    pair_sketching = sketching_features.unknown(
        sources=molecules_id.astype("uint32"),
        destinations=species_id.astype("uint32"),
        feature_combination="addition",
    )

    out = model.predict_proba(pair_sketching)
    df["proba"] = out[:, 1]

    df["note"] = df.apply(
        lambda x: check_if_in_lotus(x.species_wikidata, x.molecule_wikidata, graph),
        axis=1,
    )

    final_df = df[df.proba > 0.75].sort_values("proba", ascending=False)
    return final_df


# apply a function that looks at the species, anf if the molecule id is in the neighborhood of the species,
# then it should write in a new column called "note" this : "The link between this molecules and this species is already in LOTUS."
def check_if_in_lotus(species, molecule, graph: Graph):
    if molecule in graph.get_neighbour_node_names_from_node_name(species):
        return "This link is already in LOTUS."
    return np.nan
