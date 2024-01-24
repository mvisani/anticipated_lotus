import grape
from grape import Graph
from tqdm.auto import trange
from .external_holdouts import internal_holdout, external_holdout
from typing import Type
from .models.abstract_model import AbstractModel


def run(
    graph: Graph,
    number_of_holdouts: int,
    number_of_hops: int,
    combination: str,
    normalize: bool,
    params: dict,
    model_class: Type[AbstractModel],
):
    graph_with_only_in_taxon = graph.filter_from_names(
        edge_type_names_to_keep=["biolink:in_taxon"],
    )
    graph_without_in_taxon = graph.filter_from_names(
        edge_type_names_to_remove=["biolink:in_taxon"],
    )
    for external_holdout_number in trange(number_of_holdouts, desc="External holdouts"):
        external_holdout(
            graph_without_in_taxon,
            graph_with_only_in_taxon,
            number_of_hops=number_of_hops,
            external_holdout_number=external_holdout_number,
            combination=combination,
            normalize=normalize,
            params=params,
            model_class=model_class,
        )
