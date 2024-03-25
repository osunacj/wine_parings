from typing import List
import networkx as nx
from pyvis.network import Network
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from generate_embeddings import get_wine_dataframe
from generate_pairings import get_descriptor_frequencies


map_columns_to_relationship = {
    "variety": {
        "category": "type_of",
        "variety_location": "variety_location",
    },
    "wine": {
        "alcohol": "has_alcohol",
        "price": "has_price_of",
        "variety": "has_variety",
        "vintage": "vintage_year_of",
        "winery": "produced_by_winery",
    },
    "province": {
        "country": "in_country",
    },
    "region": {"province": "in_province"},
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
        "first": "has_aroma_of",
        "second": "has_aroma_of",
        "third": "has_aroma_of",
        "fourth": "has_aroma_of",
        "fifth": "has_aroma_of",
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


def create_triplets_from_instance(
    instance, mapper: dict, special_description: str = "wine", for_model: bool = True
):
    triplets = ""
    meta_data = {}
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

            if for_model:
                triplet = f"The object {special_description}, **{source_value}** has a relationship of **{relation}** with the object **{target_value}\n"
            else:
                triplet = f"{source_value}**{relation}**{target_value}\n"

            triplets += triplet
            meta_data[target] = target_value

    if for_model:
        return triplets, meta_data
    else:
        return triplets


def create_triplets(
    dataframe: pd.DataFrame,
    relationships: dict,
    special_description: str,
    for_model: bool = True,
):
    if for_model:
        dataframe[["triplets", "meta_data"]] = dataframe.apply(
            create_triplets_from_instance,
            args=(relationships, special_description, for_model),
            axis=1,
            result_type="expand",
        )

        return dataframe[["triplets", "meta_data"]]
    else:
        dataframe["triplets"] = dataframe.apply(
            create_triplets_from_instance,
            args=(relationships, special_description, for_model),
            axis=1,
        )
        return dataframe[["triplets"]]


mappings = {"ChÃ": "Chateau", "MarquÃÂ©s": "Marques"}


def clean_name(instance):

    if instance[:3] == "ChÃ":
        name_as_list = instance.split(" ")
        name_as_list[0] = "Chateau"
        name_str = " ".join(name_as_list)
        return name_str.strip()

    else:
        return instance


def create_wine_triplets(
    relationships=map_columns_to_relationship, n=5000, for_model: bool = True
) -> pd.DataFrame:
    wine_triplets_path = Path("./app/data/kg_triplets/wine_triplets.csv")

    # if wine_triplets_path.exists():
    #     triplets = pd.read_csv(wine_triplets_path, index_col="Unnamed: 0")
    #     return triplets

    wine_dataframe = get_wine_dataframe()
    # REMOVE THIS
    wine_dataframe = wine_dataframe.sample(n=n, axis=0, random_state=43)
    wine_dataframe.columns = map(str.lower, wine_dataframe.columns)
    wine_dataframe.rename(inplace=True, columns={"name": "wine"})
    wine_dataframe["alcohol"] = wine_dataframe["alcohol"].apply(
        lambda x: np.nan if str(x) == "NaN" or str(x) == "nan" else x
    )
    wine_dataframe["wine"] = wine_dataframe["wine"].apply(clean_name)
    wine_dataframe["alcohol"].fillna(wine_dataframe["alcohol"].mode(), inplace=True)
    wine_dataframe["vintage"].fillna(wine_dataframe["vintage"].mean(), inplace=True)
    wine_dataframe["price"] = wine_dataframe["price"].apply(
        lambda x: np.nan if str(x) == "NaN" or str(x) == "nan" else x
    )
    wine_dataframe["price"].fillna(wine_dataframe["price"].mode(), inplace=True)
    wine_dataframe["vintage"] = wine_dataframe["vintage"].apply(lambda x: int(x))
    wine_dataframe["variety_location"] = (
        wine_dataframe["variety"] + " " + wine_dataframe["geo_normalized"]
    )
    triplets = create_triplets(
        wine_dataframe, relationships, special_description="wine", for_model=for_model
    )

    # triplets.to_csv(wine_triplets_path)

    return triplets


def create_variety_descriptor_triplets(
    relationships=descriptor_relationship, for_model: bool = True
) -> pd.DataFrame:
    variety_descriptors = get_descriptor_frequencies()
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

    return create_triplets(
        variety_descriptors,
        relationships,
        special_description="variety",
        for_model=for_model,
    )


def create_food_triplets(
    relationships=recipe_relationships, for_model: bool = True
) -> pd.DataFrame:
    recipe_pairings_path = Path("./app/data/production/recipe_pairings.csv")
    if recipe_pairings_path.exists():
        recipe_pairings = pd.read_csv(recipe_pairings_path, index_col="index")

    return create_triplets(
        recipe_pairings, relationships, special_description="dish", for_model=for_model
    )


def create_kg_triplets(sample_size=1000, for_model: bool = True):
    KG = pd.concat(
        [
            create_food_triplets(for_model=for_model),
            create_wine_triplets(n=5000, for_model=for_model),
            create_variety_descriptor_triplets(for_model=for_model),
        ],
        axis=0,
    )

    return KG.sample(n=sample_size, axis=0, random_state=43)


def main():

    KG = create_kg_triplets(sample_size=50)

    # G = nx.DiGraph()
    # for _, row in KG.iterrows():
    #     G.add_edge(row["head"], row["tail"], label=row["edges"])
    #     # G.nodes[row['tail']]['label'] = row['node_label']

    # pos = nx.spring_layout(G, seed=42, k=1)
    # labels = nx.get_edge_attributes(G, "label")
    # plt.figure(figsize=(12, 10))
    # nx.draw(
    #     G,
    #     pos,
    #     font_size=10,
    #     node_size=700,
    #     node_color="lightblue",
    #     edge_color="gray",
    #     alpha=0.6,
    # )
    # nx.draw_networkx_edge_labels(
    #     G,
    #     pos,
    #     edge_labels=labels,
    #     font_size=8,
    #     label_pos=0.3,
    #     verticalalignment="baseline",
    # )
    # plt.title("Knowledge Graph")
    # plt.show()

    # net = Network(
    #     notebook=True,
    #     cdn_resources="remote",
    #     bgcolor="#222222",
    #     font_color="white",
    #     height="750px",
    #     width="100%",
    #     select_menu=True,
    #     filter_menu=True,
    # )
    # net.show_buttons(filter_="physics")
    # net.from_nx(G)
    # net.show("nx.html")


if __name__ == "__main__":
    main()
