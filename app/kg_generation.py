from distutils.command.build_scripts import first_line_re
import imp
import networkx as nx
from pyvis.network import Network
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from generate_embeddings import get_wine_dataframe
from generate_pairings import get_descriptor_frequencies


map_columns_to_relationship = {
    "variety": {"category": "from_category", "variety_location": "variety_location"},
    "wine": {
        "alcohol": "has_alcohol",
        "category": "has_category",
        "price": "has_price",
        "variety": "has_variety",
        "vintage": "vintage_year",
        "winery": "from_winery",
    },
    "province": {
        "country": "from_country",
    },
    "region": {"province": "from_province"},
    "winery": {
        "province": "from_province",
        "region": "from_region",
        "country": "from_country",
        "variety_location": "produces_variety",
        # "geo_normalized": "from_location"
    },
}

descriptor_relationship = {
    "variety_location": {
        "first": "has_aroma",
        "second": "has_aroma",
        "third": "has_aroma",
        "fourth": "has_aroma",
        "fifth": "has_aroma",
    }
}

recipe_relationships = {
    "name": {
        "first_pairing": "pairs_with",
        "second_pairing": "pairs_with",
        "third_pairing": "pairs_with",
        "fourth_pairing": "pairs_with",
    }
}


skip_columns = ["0"]


def create_nodes_from_instance(instance, mapper: dict):
    instance_sources = []
    instance_targets = []
    instance_source_labels = []
    instance_target_labels = []
    relations = []
    for source, targets in mapper.items():
        source_value = instance[source]
        source_label = source

        if source_value is np.nan or source_value == "none":
            continue

        for target, relation in targets.items():
            target_value = instance[target]
            target_label = target

            if target_value is np.nan or target_value == "none":
                continue

            relations.append(relation)
            instance_sources.append(str(source_value))
            instance_targets.append(target_value)
            instance_source_labels.append(source_label)
            instance_target_labels.append(target_label)

    return (
        instance_sources,
        instance_targets,
        relations,
        instance_source_labels,
        instance_target_labels,
    )


def create_triplets(dataframe: pd.DataFrame, relationships: dict):
    triplets = {"heads": [], "tails": [], "edges": [], "h_labels": [], "t_labels": []}

    # def extend_nodes():

    for _, instance in dataframe.iterrows():
        (
            instance_sources,
            instance_targets,
            relations,
            instance_source_labels,
            instance_target_labels,
        ) = create_nodes_from_instance(instance, mapper=relationships)
        triplets["heads"].extend(instance_sources)
        triplets["edges"].extend(relations)
        triplets["tails"].extend(instance_targets)
        triplets["h_labels"].extend(instance_source_labels)
        triplets["t_labels"].extend(instance_target_labels)

    return pd.DataFrame(
        {
            "head": triplets["heads"],
            "tail": triplets["tails"],
            "edges": triplets["edges"],
        }
    )


def create_wine_triplets(relationships=map_columns_to_relationship) -> pd.DataFrame:
    wine_dataframe = get_wine_dataframe()
    # REMOVE THIS
    wine_dataframe = wine_dataframe.sample(n=50, axis=0, random_state=43)
    wine_dataframe.columns = map(str.lower, wine_dataframe.columns)
    wine_dataframe.rename(inplace=True, columns={"name": "wine"})
    wine_dataframe["vintage"].fillna(wine_dataframe["vintage"].mean(numeric_only=True))
    wine_dataframe["price"].fillna(wine_dataframe["price"].mode())
    wine_dataframe["vintage"] = wine_dataframe["vintage"].apply(lambda x: int(x))
    wine_dataframe["variety_location"] = (
        wine_dataframe["variety"] + " " + wine_dataframe["geo_normalized"]
    )
    return create_triplets(wine_dataframe, relationships)


def create_variety_descriptor_triplets(
    relationships=descriptor_relationship,
) -> pd.DataFrame:
    variety_descriptors = get_descriptor_frequencies()
    variety_descriptors = variety_descriptors.sample(n=50, axis=0, random_state=43)
    columns = variety_descriptors.columns
    for column in columns:
        if column != "descriptors":
            variety_descriptors.drop(column, inplace=True, axis=1)

    variety_descriptors.columns = map(str.lower, variety_descriptors.columns)
    variety_descriptors.reset_index(inplace=True)
    variety_descriptors.rename({"index": "variety_location"}, axis=1, inplace=True)
    variety_descriptors["descriptors"] = variety_descriptors["descriptors"].apply(
        lambda x: ",".join(eval(x))
    )
    variety_descriptors[["first", "second", "third", "fourth", "fifth"]] = (
        variety_descriptors["descriptors"].str.split(",", expand=True)
    )
    variety_descriptors.drop(["descriptors"], inplace=True, axis=1)

    return create_triplets(variety_descriptors, relationships)


def create_food_triplets(relationships=recipe_relationships) -> pd.DataFrame:
    recipe_pairings_path = Path("./app/data/production/recipe_pairings.csv")
    if recipe_pairings_path.exists():
        recipe_pairings = pd.read_csv(recipe_pairings_path, index_col="index")

    return create_triplets(recipe_pairings, relationships)


def main():

    # G=nx.from_pandas_edgelist(KG, "head", "tail", edge_key = 'labels', create_using=nx.MultiDiGraph())

    KG = pd.concat(
        [create_variety_descriptor_triplets(), create_wine_triplets()], axis=0
    )

    G = nx.DiGraph()
    for _, row in KG.iterrows():
        G.add_edge(row["head"], row["tail"], label=row["edges"])
        # G.nodes[row['tail']]['label'] = row['node_label']

    pos = nx.spring_layout(G, seed=42, k=1)
    labels = nx.get_edge_attributes(G, "label")
    plt.figure(figsize=(12, 10))
    nx.draw(
        G,
        pos,
        font_size=10,
        node_size=700,
        node_color="lightblue",
        edge_color="gray",
        alpha=0.6,
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=labels,
        font_size=8,
        label_pos=0.3,
        verticalalignment="baseline",
    )
    plt.title("Knowledge Graph")
    plt.show()

    net = Network(
        notebook=True,
        cdn_resources="remote",
        bgcolor="#222222",
        font_color="white",
        height="750px",
        width="100%",
        select_menu=True,
        filter_menu=True,
    )
    net.show_buttons(filter_="physics")
    net.from_nx(G)
    net.show("nx.html")


if __name__ == "__main__":
    main()
