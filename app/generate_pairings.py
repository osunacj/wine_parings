import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy import spatial

from notebooks.helpers.prep.pairing_rules import (
    nonaroma_rules,
    congruent_or_contrasting,
    sort_by_aroma_similarity,
    retrieve_pairing_type_info,
)


def get_production_wines():
    wines_df = pd.read_csv(
        "./app/data/production/wine_production.csv", index_col="Unnamed: 0"
    )
    return wines_df


def get_descriptor_frequencies():
    descriptor_frequencies = pd.read_csv(
        "./app/notebooks/helpers/temp/wine_variety_descriptors.csv", index_col="index"
    )
    # This file is still incorrect and it needs to be fixed
    return descriptor_frequencies


def get_food_embedding_dict():
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )

    with food_to_embeddings_dict_path.open("rb") as f:
        food_to_embeddings_dict = pickle.load(f)

    return food_to_embeddings_dict


def get_average_nonaroma_embeddings():
    food_similarity_dict_path = Path(
        f"./app/notebooks/helpers/models/food_similarity_dict.pkl"
    )

    with food_similarity_dict_path.open("rb") as f:
        food_nonaroma_infos = pickle.load(f)
    return food_nonaroma_infos


def normalize_production_wines():
    wines_df = get_production_wines()
    wine_weights = wine_weights = {
        "weight": {1: (0, 0.25), 2: (0.25, 0.45), 3: (0.45, 0.75), 4: (0.75, 1)},
        "sweet": {1: (0, 0.25), 2: (0.25, 0.6), 3: (0.6, 0.75), 4: (0.75, 1)},
        "acid": {1: (0, 0.05), 2: (0.05, 0.25), 3: (0.25, 0.5), 4: (0.5, 1)},
        "salt": {1: (0, 0.15), 2: (0.15, 0.25), 3: (0.25, 0.7), 4: (0.7, 1)},
        "piquant": {1: (0, 0.15), 2: (0.15, 0.3), 3: (0.3, 0.6), 4: (0.6, 1)},
        "fat": {1: (0, 0.25), 2: (0.25, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
        "bitter": {1: (0, 0.2), 2: (0.2, 0.37), 3: (0.37, 0.6), 4: (0.6, 1)},
    }

    for taste, subdict in wine_weights.items():
        wines_df[taste] = wines_df[taste].apply(lambda x: check_in_range(subdict, x))
    wines_df.sort_index(inplace=True)
    return wines_df


# this function scales each nonaroma between 0 and 1
def minmax_scaler(val, minval, maxval):
    val = max(min(val, maxval), minval)
    normalized_val = (val - minval) / (maxval - minval)
    return normalized_val


# this function makes sure that a scaled value (between 0 and 1) is returned for a food nonaroma
def check_in_range(taste_weights, value):
    for label, value_range_tuple in taste_weights.items():
        lower_end = value_range_tuple[0]
        upper_end = value_range_tuple[1]
        if value >= lower_end and value <= upper_end:
            return label
        else:
            continue


# this function calculates the average embedding of all foods supplied as input
def calculate_avg_food_vec(food_ingredients: list):
    food_to_embeddings_dict = get_food_embedding_dict()
    sample_food_vecs = []
    for ingredient in food_ingredients:
        if ingredient in food_to_embeddings_dict:
            sample_food_vec = food_to_embeddings_dict[ingredient]
            # Get the average embedding of the same ingredient coming from the food dict
            sample_food_vecs.append(np.average(sample_food_vec, axis=0))
    # Get the average of all the ingredients in the food.
    sample_food_vecs_avg = np.average(sample_food_vecs, axis=0)
    return sample_food_vecs_avg


# this function returns two things: a score (between 0 and 1) and a normalized value (integer between 1 and 4) for a given nonaroma
def calculate_food_attributes(average_food_embedding):
    tastes_weights = {
        "weight": {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.7), 4: (0.7, 1)},
        "sweet": {1: (0, 0.45), 2: (0.45, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
        "acid": {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
        "salt": {1: (0, 0.3), 2: (0.3, 0.55), 3: (0.55, 0.8), 4: (0.8, 1)},
        "piquant": {1: (0, 0.4), 2: (0.4, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
        "fat": {1: (0, 0.4), 2: (0.4, 0.5), 3: (0.5, 0.6), 4: (0.6, 1)},
        "bitter": {1: (0, 0.3), 2: (0.3, 0.5), 3: (0.5, 0.65), 4: (0.65, 1)},
    }
    food_nonaroma_embeddings = get_average_nonaroma_embeddings()
    food_attributes = dict()
    for taste in ["weight", "sweet", "acid", "salt", "piquant", "fat", "bitter"]:
        similarity = 1 - spatial.distance.cosine(
            food_nonaroma_embeddings[taste]["average_vec"], average_food_embedding
        )  # type: ignore
        # scale the similarity using our minmax scaler
        scaled_similarity = minmax_scaler(
            similarity,
            food_nonaroma_embeddings[taste]["farthest"],
            food_nonaroma_embeddings[taste]["closest"],
        )
        standardized_similarity = check_in_range(
            tastes_weights[taste], scaled_similarity
        )
        food_attributes[taste] = (scaled_similarity, standardized_similarity)
    return food_attributes


def get_food_attributes(food_ingredients):
    average_food_embedding = calculate_avg_food_vec(food_ingredients)
    food_attributes = calculate_food_attributes(average_food_embedding)
    return food_attributes, average_food_embedding


def get_pairings_type(wine_recommendations):
    try:
        (
            contrasting_wines,
            contrasting_nonaromas,
            contrasting_body,
            # impactful_descriptors_contrasting,
        ) = retrieve_pairing_type_info(wine_recommendations, "contrasting")
    except:
        contrasting_wines = []

    try:
        (
            congruent_wines,
            congruent_nonaromas,
            congruent_body,
            impactful_descriptors_congruent,
        ) = retrieve_pairing_type_info(wine_recommendations, "congruent")
    except:
        congruent_wines = []


def main():
    hotdog = [
        "hotdog",
        "mustard",
        "tomato",
        "onion",
        "pepperoncini",
        "gherkin",
        "celery",
        "relish",
    ]
    salmon = ["smoked_salmon", "dill", "cucumber", "sour_cream"]
    food_attributes, average_food_embedding = get_food_attributes(salmon)

    wine_recommendations = normalize_production_wines()
    wine_recommendations = nonaroma_rules(wine_recommendations, food_attributes)
    wine_recommendations = congruent_or_contrasting(
        wine_recommendations, food_attributes
    )
    wine_recommendations = sort_by_aroma_similarity(
        wine_recommendations, average_food_embedding
    )
    # wine_recommendations["most_impactful_descriptors"] = wine_recommendations.index.map(
    #     most_impactful_descriptors
    # )

    # see if there are any contrasting suggestions


if __name__ == "__main__":
    main()
