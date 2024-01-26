import pandas as pd
import grape
from grape import Graph
from .external_holdouts import external_holdout
from tqdm.auto import trange
from typing import Type
from .models.abstract_model import AbstractModel
from cache_decorator import Cache
from deflate_dict import deflate


@Cache(
    cache_dir="experiments/n_hops_{number_of_hops}/combination_{combination}/normalize_{normalize}/{_hash}",
    cache_path="{cache_dir}/performance.csv",
)
def run(
    graph: Graph,
    number_of_external_holdouts: int,
    number_of_internal_holdouts: int,
    number_of_hops: int,
    combination: str,
    normalize: bool,
    model_class: Type[AbstractModel],
    max_evals: int,
):
    graph_with_only_in_taxon = graph.filter_from_names(
        edge_type_names_to_keep=["biolink:in_taxon"],
    )
    graph_without_in_taxon = graph.filter_from_names(
        edge_type_names_to_remove=["biolink:in_taxon"],
    )

    results = []
    for external_holdout_number in trange(
        number_of_external_holdouts,
        desc="External holdouts",
        leave=False,
    ):
        tmp = deflate(
            external_holdout(
                graph_without_in_taxon,
                graph_with_only_in_taxon,
                number_of_hops,
                external_holdout_number,
                number_of_internal_holdouts,
                combination,
                normalize,
                model_class,
                max_evals=max_evals,
            )
        )
        tmp["external_holdout_number"] = external_holdout_number
        tmp["number_of_hops"] = number_of_hops
        tmp["combination"] = combination
        tmp["normalize"] = normalize
        tmp["model_class"] = model_class.__name__
        tmp["max_evals"] = max_evals

        results.append(tmp)

    return pd.DataFrame(results)
