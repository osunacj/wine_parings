import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, List, Dict
from collections import defaultdict, Counter
from scipy import spatial
from sklearn.decomposition import PCA
import random
import torch
from tqdm import tqdm
import pickle

from notebooks.helpers.models.embedding_model import PredictionModel
from notebooks.helpers.prep.utils import (
    get_all_ingredients,
    modify_vocabulary,
    _random_sample_with_min_count,
    _merge_synonmys,
    generate_average_PCA_from_embeddings,
)
from notebooks.helpers.prep.wine_mapping_values import (
    wine_terms_mappings,
    food_taste_mappings,
)
from notebooks.helpers.prep.ingredients_mapping import ingredients_mappings


core_tastes = [
    "aroma",
    "weight",
    "sweet",
    "acid",
    "salt",
    "piquant",
    "fat",
    "bitter",
    "flavor",
]

all_ingredients = get_all_ingredients(as_list=True)
food_terms_map = {**food_taste_mappings}


def get_wine_dataframe():
    wine_dataset = pd.read_csv("./app/data/test/reduced_wines.csv", encoding="latin-1")
    wine_dataset.drop(["index"], inplace=True, axis=1)
    wine_dataset.reset_index(inplace=True, drop=True)
    return wine_dataset


def get_food_dataframe():
    food_dataset = pd.read_csv("./app/data/test/reduced_food.csv")
    food_dataset.reset_index(inplace=True, drop=True)
    return food_dataset


def _generate_food_sentence_dict():
    wine_dataset = get_wine_dataframe()
    wine_reviews = wine_dataset["clean_descriptions"].to_numpy()
    ingredients_in_reviews = wine_dataset["descriptors_in_reviews"].to_numpy()

    food_dataset = get_food_dataframe()
    food_instructions = food_dataset["clean_instructions"].to_numpy()
    ingredients_in_instructions = food_dataset["ingredients_in_instructions"]

    # add food reviews here
    instruction_sentences = np.concatenate((wine_reviews, food_instructions))
    ingredients_in_sentences = np.concatenate(
        (ingredients_in_reviews, ingredients_in_instructions)
    )

    food_to_sentences_dict = defaultdict(list)
    for sentence, ingredients_in_sentence in zip(
        instruction_sentences, ingredients_in_sentences
    ):
        for ingredient in eval(ingredients_in_sentence):
            if ingredient in all_ingredients and ingredient in sentence:
                food_to_sentences_dict[ingredient].append(
                    sentence.replace(".", " . ").strip()
                )

    return food_to_sentences_dict


def sample_random_sentence_dict(max_sentence_count):
    food_to_sentences_dict = _generate_food_sentence_dict()
    # only keep 100 randomly selected sentences
    food_to_sentences_dict_random_samples = {
        food: _random_sample_with_min_count(sentences, max_sentence_count)
        for food, sentences in food_to_sentences_dict.items()
    }
    return food_to_sentences_dict_random_samples


def _map_ingredients_to_input_ids():
    model = PredictionModel()
    ingredient_ids = model.tokenizer.convert_tokens_to_ids(all_ingredients)  # type: ignore
    ingredient_ids_dict = dict(zip(all_ingredients, ingredient_ids))  # type: ignore
    return ingredient_ids_dict


def generate_food_embedding_dict(max_sentence_count=100, force=False):
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )
    if food_to_embeddings_dict_path.exists() and not force:
        with food_to_embeddings_dict_path.open("rb") as f:
            food_to_embeddings_dict = pickle.load(f)

        # delete keys if we deleted ingredients
        old_ingredients = set(food_to_embeddings_dict.keys())
        new_ingredients = set(get_all_ingredients())

        keys_to_delete = old_ingredients.difference(new_ingredients)
        for key in keys_to_delete:
            food_to_embeddings_dict.pop(key, None)  # delete key if it exists

        food_to_embeddings_dict = _merge_synonmys(food_to_embeddings_dict)

        with food_to_embeddings_dict_path.open("wb") as f:
            pickle.dump(
                food_to_embeddings_dict, f
            )  # Overwrite dict with cleaned version

    else:
        print("\nSampling Random Sentences")
        food_to_sentences_dict_random_samples = sample_random_sentence_dict(
            max_sentence_count=max_sentence_count
        )
        food_to_embeddings_dict = defaultdict(list)
        print("\nMapping Ingredients to Input Ids")
        all_ingredient_ids = _map_ingredients_to_input_ids()

        prediction_model = PredictionModel()
        for food, sentences in tqdm(
            food_to_sentences_dict_random_samples.items(),
            total=len(food_to_sentences_dict_random_samples),
            desc="Calculating Embeddings for Food items",
        ):
            embeddings, ingredient_ids = prediction_model.predict_embeddings(sentences)
            # get embedding of food word
            embeddings_flat = embeddings.view((-1, 768))
            ingredient_ids_flat = torch.stack(ingredient_ids).flatten()
            food_id = all_ingredient_ids[food]
            food_embeddings = (
                embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()
            )
            food_to_embeddings_dict[food].extend(food_embeddings)

        food_to_embeddings_dict = {
            k: np.stack(v) for k, v in food_to_embeddings_dict.items() if v
        }

        food_to_embeddings_dict = _merge_synonmys(food_to_embeddings_dict)

        with food_to_embeddings_dict_path.open("wb") as f:
            pickle.dump(food_to_embeddings_dict, f)

    return food_to_embeddings_dict


def get_taste_of_ingredient(ingredient):
    ingredient = ingredient.replace("_", " ")
    taste = "flavor"

    if ingredient in wine_terms_mappings:
        taste = wine_terms_mappings[ingredient][2]

    return taste


def get_top_words_in_variety(wines_in_variety, taste, n=50):
    # write the descriptors for each variety, geo, taste -> all descritors of aroma
    all_descriptors = []
    for descriptors in list(wines_in_variety[taste + "_descriptors"]):
        if descriptors is np.nan:
            continue

        for descriptor in descriptors:
            if len(descriptor) > 1:
                all_descriptors.append(descriptor)

    descriptor_frequencies = Counter(all_descriptors)
    # Get most common word frequencies
    most_common_words = descriptor_frequencies.most_common(n)
    top_n_words = [
        (i[0], "{:.1f}".format(i[1])) for i in most_common_words if len(i[0]) > 2
    ]
    return top_n_words


def normalize(df, cols_to_normalize):
    for feature_name in cols_to_normalize:
        max_value = df[feature_name].max()
        min_value = df[feature_name].min()
        df[feature_name] = df[feature_name].apply(
            lambda x: (x - min_value) / (max_value - min_value)
        )
    return df


def average_taste_embeddings(dataframe, force):
    wine_average_embeddings_path = Path(
        "./app/notebooks/helpers/models/wine_average_embeddings.pkl"
    )
    if wine_average_embeddings_path.exists() and not force:
        with wine_average_embeddings_path.open("rb") as f:
            wine_average_embeddings = pickle.load(f)
            return wine_average_embeddings

    # pull the average embedding for the wine attribute across all wines.
    wine_average_embeddings = dict()
    for core_taste in core_tastes:
        # look at the average embedding for a taste, across all wines that have descriptors for that taste
        review_arrays = dataframe[core_taste + "_embeddings"].dropna()
        # Average of the embeddings for a given tate
        wine_average_embeddings[core_taste] = np.average(review_arrays, axis=0)

    with wine_average_embeddings_path.open("wb") as f:
        pickle.dump(wine_average_embeddings, f)

    return wine_average_embeddings


def construct_taste_ingredient_embeddings(
    food_to_embeddings_dict, descriptors_in_reviews
):
    constructor = {}
    for core_taste in core_tastes:
        constructor[core_taste + "_descriptors"] = []
        constructor[core_taste + "_embeddings"] = []

    for review in tqdm(
        descriptors_in_reviews,
        total=len(descriptors_in_reviews),
        desc="Ingredient By Taste Embeddings",
    ):
        for core_taste in core_tastes:
            embeddings = []
            ingredients = []
            for ingredient in eval(review):
                taste = get_taste_of_ingredient(ingredient)
                if ingredient not in food_to_embeddings_dict:
                    avg_embedding = np.nan
                else:
                    avg_embedding = np.average(
                        food_to_embeddings_dict.get(ingredient), axis=0  # type: ignore
                    )  # type: ignore
                if taste == core_taste:
                    ingredients.append(ingredient)
                    embeddings.append(avg_embedding)
            embeddings = list(filter(lambda x: x is not np.nan, embeddings))
            constructor[core_taste + "_embeddings"].append(
                # Average embedding of taste for review
                np.average(embeddings, axis=0)
                if embeddings
                else np.nan
            )
            constructor[core_taste + "_descriptors"].append(ingredients)
    embeddings_dataset = pd.DataFrame(constructor, columns=list(constructor.keys()))
    return embeddings_dataset


def wine_varieties(food_to_embeddings_dict, force=False):
    wine_dataset = get_wine_dataframe()
    descriptors_in_reviews = wine_dataset["descriptors_in_reviews"].to_numpy()
    wine_embeddings_dataframe = construct_taste_ingredient_embeddings(
        food_to_embeddings_dict, descriptors_in_reviews
    )
    dataframe = pd.concat(
        [wine_dataset, wine_embeddings_dataframe], axis=1
    ).reset_index()
    avg_taste_embeddings = average_taste_embeddings(dataframe, force=False)

    wine_varieties = list(set(zip(dataframe["Variety"], dataframe["geo_normalized"])))
    wine_varieties = list(
        filter(lambda x: type(x[0]) == str and type(x[1]) == str, wine_varieties)
    )

    wine_varieties_index = [f"{variety} {geo}" for variety, geo in wine_varieties]

    taste_variety_dataframes = []
    average_variety_vec = []

    for taste in core_tastes:
        wine_variety_vectors = {}
        wine_variety_top_words = []
        for variety in wine_varieties:
            wines_in_variety = dataframe.loc[
                (dataframe["Variety"] == variety[0])
                & (dataframe["geo_normalized"] == variety[1])
            ]

            if wines_in_variety.shape[0] < 1 or str(variety[1][-1]) == "0":
                continue

            # if vector exits place existent vector otherwise place average of taste (wine attribute)
            taste_embeddings = [
                (
                    embedding
                    if type(embedding) == np.ndarray
                    else avg_taste_embeddings[taste]
                )
                for embedding in wines_in_variety[taste + "_embeddings"].to_numpy()
            ]

            if taste == "aroma":
                top_n_words = get_top_words_in_variety(wines_in_variety, taste, n=15)
                wine_variety_top_words.append(top_n_words)

            wine_variety_vectors[variety] = taste_embeddings
            # average vector of the wine for a given taste and variety

        if taste not in ["aroma", "flavor"]:
            ordered_pca_variety = generate_average_PCA_from_embeddings(
                wine_varieties, wine_variety_vectors, N=1
            )
            wine_variety_vectors = pd.DataFrame(
                ordered_pca_variety,
                index=wine_varieties_index,
                columns=[f"{taste}"],
            )
        else:
            ordered_pca_variety = generate_average_PCA_from_embeddings(
                wine_varieties, wine_variety_vectors, N=0
            )
            wine_variety_vectors = pd.Series(
                ordered_pca_variety, index=wine_varieties_index, name=f"{taste}"
            )
            if taste == "aroma":
                wine_variety_descriptions = pd.DataFrame(
                    wine_variety_top_words, index=wine_varieties_index
                )
        wine_variety_vectors.sort_index(inplace=True)
        taste_variety_dataframes.append(wine_variety_vectors)

    wine_variety_descriptions.to_csv(
        "./app/notebooks/helpers/temp/wine_variety_descriptors.csv"
    )
    wines = pd.concat(taste_variety_dataframes, axis=1, ignore_index=False)
    wines_norm = normalize(wines, cols_to_normalize=core_tastes[1:-1])
    wines_norm.to_csv("./app/data/production/wine_production.csv")


def generate_similiarity_dict(food_to_embeddings_dict, force=False):
    food_similarity_dict_path = Path(
        f"./app/notebooks/helpers/models/food_similarity_dict.pkl"
    )

    food_tastes_dict_path = Path(f"./app/notebooks/helpers/models/food_taste_dict.pkl")

    if food_tastes_dict_path.exists() and not force:
        with food_tastes_dict_path.open("rb") as f:
            core_tastes_distances = pickle.load(f)
            return core_tastes_distances

    if food_similarity_dict_path.exists() and not force:
        with food_similarity_dict_path.open("rb") as f:
            food_nonaroma_infos = pickle.load(f)
            return food_nonaroma_infos

    # Compute the average taste distances
    average_taste_embedding_new = dict()
    for taste, food_items in food_taste_mappings.items():
        taste_avg_embedding = []
        for food in food_items:
            food = food.replace(" ", "_")
            if food in food_to_embeddings_dict:
                embedding_avg = food_to_embeddings_dict.get(food)
                embedding_avg = np.average(embedding_avg, axis=0)
                taste_avg_embedding.append(embedding_avg)

        average_taste_embedding_new[taste] = np.average(taste_avg_embedding, axis=0)

    # Compute the distance of every word to the average distance of every taste
    avg_taste_embeddings = average_taste_embedding_new
    core_tastes_distances = dict()
    for taste in core_tastes:
        tastes_distances = dict()
        for food in ingredients_mappings.values():
            food = food.replace(" ", "_")
            if food in food_to_embeddings_dict:
                embedding = food_to_embeddings_dict.get(food)
                embedding = np.average(embedding, axis=0)  # type: ignore
                similarity = 1 - spatial.distance.cosine(
                    avg_taste_embeddings[taste],
                    # average of all the embeddings of ingredient food
                    embedding,
                )  # type: ignore
                tastes_distances[food] = similarity
        core_tastes_distances[taste] = tastes_distances

    food_nonaroma_infos = dict()
    # for each core taste, identify the food item that is farthest and closest. We will need this to create a normalized scale between 0 and 1
    for key in core_tastes:
        dict_taste = dict()
        farthest = min(core_tastes_distances[key], key=core_tastes_distances[key].get)
        farthest_distance = core_tastes_distances[key][farthest]
        closest = max(core_tastes_distances[key], key=core_tastes_distances[key].get)
        closest_distance = core_tastes_distances[key][closest]
        # Closest distance means that the vector is more similar
        print(f" Taste: {key} Farthest: {farthest} Closest: {closest}")
        dict_taste["farthest"] = farthest_distance
        dict_taste["farthest_vec"] = np.average(
            food_to_embeddings_dict.get(farthest), axis=0
        )
        dict_taste["closest_vec"] = np.average(
            food_to_embeddings_dict.get(closest), axis=0
        )
        dict_taste["closest"] = closest_distance
        dict_taste["average_vec"] = avg_taste_embeddings[key]
        food_nonaroma_infos[key] = dict_taste

    with food_tastes_dict_path.open("wb") as f:
        pickle.dump(core_tastes_distances, f)

    with food_similarity_dict_path.open("wb") as f:
        pickle.dump(food_nonaroma_infos, f)


def main():
    modify_vocabulary()
    food_to_embeddings_dict = generate_food_embedding_dict(
        max_sentence_count=100, force=False
    )
    # wine_varieties(food_to_embeddings_dict, force=True)
    generate_similiarity_dict(food_to_embeddings_dict, force=True)


if __name__ == "__main__":
    main()
