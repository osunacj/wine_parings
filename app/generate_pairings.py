import random
from typing import Union
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import pickle
from scipy import spatial
import re
import ast

from generate_embeddings import get_food_dataframe
from notebooks.helpers.models.embedding_model import PredictionModel
from notebooks.helpers.prep.wine_mapping_values import food_taste_mappings
from notebooks.helpers.prep.pairing_rules import (
    nonaroma_rules,
    congruent_or_contrasting,
    sort_by_aroma_similarity,
    retrieve_pairing_type_info,
)

from notebooks.helpers.prep.view_embeddings import (
    plot_wine_recommendations,
    view_embeddings_of_ingredient,
    view_dish_embeddings,
)

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


def get_production_wines():
    wines_df = pd.read_csv(
        "./app/data/production/wine_production.csv", index_col="Unnamed: 0"
    )
    return wines_df


def get_descriptor_frequencies():
    descriptor_frequencies = pd.read_csv(
        "./app/notebooks/helpers/temp/wine_variety_descriptors.csv",
        index_col="Unnamed: 0",
    )

    variety_descriptors_path = Path("./app/data/production/variety_descriptors.csv")

    if variety_descriptors_path.exists():
        variety_descriptors = pd.read_csv(
            variety_descriptors_path, index_col="Unnamed: 0"
        )
        return variety_descriptors

    index = descriptor_frequencies.index
    columns = descriptor_frequencies.columns
    descriptors = []

    for _, variety in descriptor_frequencies.iterrows():
        desc = []
        for column in columns[:5]:
            descriptor = variety[column]
            descriptor = eval(descriptor)
            desc.append(descriptor[0])
        descriptors.append(desc)

    desc_df = pd.DataFrame({"descriptors": descriptors}, index=index)
    wines_df = get_production_wines()
    variety_descriptors = pd.merge(desc_df, wines_df, right_index=True, left_index=True)
    variety_descriptors.to_csv(variety_descriptors_path)
    return variety_descriptors


def get_food_embedding_dict():
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )

    with food_to_embeddings_dict_path.open("rb") as f:
        food_to_embeddings_dict = pickle.load(f)

    return food_to_embeddings_dict


def get_food_taste_distances_info():
    food_similarity_dict_path = Path(
        f"./app/notebooks/helpers/models/food_similarity_dict.pkl"
    )

    food_tastes_dict_path = Path(f"./app/notebooks/helpers/models/food_taste_dict.pkl")

    with food_tastes_dict_path.open("rb") as f:
        food_tastes_distances = pickle.load(f)

    with food_similarity_dict_path.open("rb") as f:
        food_average_distances = pickle.load(f)
    return food_average_distances


def nparray_str_to_list(array_string):
    average_taste_vec = re.sub("\s+", ",", array_string)
    average_taste_vec = average_taste_vec.replace("[,", "[")
    average_taste_vec = np.array(ast.literal_eval(average_taste_vec))
    return average_taste_vec


def normalize_production_wines():
    wine_weights = {
        "weight": {1: (0, 0.25), 2: (0.25, 0.50), 3: (0.50, 0.75), 4: (0.75, 1)},
        "sweet": {1: (0, 0.15), 2: (0.15, 0.2), 3: (0.2, 0.4), 4: (0.4, 1)},
        "acid": {1: (0, 0.2), 2: (0.2, 0.3), 3: (0.3, 0.4), 4: (0.4, 1)},
        "salt": {1: (0, 0.55), 2: (0.55, 0.6), 3: (0.6, 0.7), 4: (0.7, 1)},
        "piquant": {1: (0, 0.2), 2: (0.2, 0.3), 3: (0.3, 0.4), 4: (0.4, 1)},
        "fat": {1: (0, 0.25), 2: (0.25, 0.3), 3: (0.3, 0.4), 4: (0.4, 1)},
        "bitter": {1: (0, 0.12), 2: (0.12, 0.3), 3: (0.3, 0.5), 4: (0.5, 1)},
    }

    wines = get_descriptor_frequencies()
    wines["aroma"] = wines["aroma"].apply(nparray_str_to_list)
    wines["flavor"] = wines["flavor"].apply(nparray_str_to_list)

    for taste, subdict in wine_weights.items():
        wines[taste] = wines[taste].apply(lambda x: check_in_range(subdict, x))
    wines.sort_index(inplace=True)
    return wines


# this function scales each nonaroma between 0 and 1
def minmax_scaler(val, minval, maxval):
    val = max(min(val, maxval), minval)
    normalized_val = (val - minval) / (maxval - minval)
    return normalized_val


# this function makes sure that a scaled value (between 0 and 1) is returned for a food nonaroma
def check_in_range(taste_weights, value):
    for label, value_range_tuple in taste_weights.items():
        lower_end = value_range_tuple[0]
        upper_end = value_range_tuple[1] + 0.01
        if value >= lower_end and value < upper_end:
            return label
        else:
            continue


# this function calculates the average embedding of all foods supplied as input
def calculate_avg_food_vec(
    food_average_distances,
    dish_ingredients: dict,
) -> dict:
    ingredients_tastes = {}
    for core_taste in core_tastes:
        sample_food_vecs = []
        # thresh = (
        #     food_average_distances[core_taste]["closest"]
        #     - (
        #         food_average_distances[core_taste]["closest"]
        #         - food_average_distances[core_taste]["farthest"]
        #     )
        #     / 2
        # )
        # taste_avg = food_average_distances[core_taste]["average_vec"]

        for ingredient in dish_ingredients.keys():
            food_embedding = dish_ingredients[ingredient]
            if core_taste == "aroma":
                sample_food_vecs.append(food_embedding)
            else:
                if ingredient in food_taste_mappings[core_taste]:
                    sample_food_vecs.append(food_embedding)

        if len(sample_food_vecs) > 0:
            embed = np.average(sample_food_vecs, axis=0)
        else:
            embed = ingredients_tastes.get("aroma", [])
            if embed == []:
                return {}

        ingredients_tastes[core_taste] = embed
    return ingredients_tastes


# this function returns two things: a score (between 0 and 1) and a normalized value (integer between 1 and 4) for a given nonaroma
def calculate_food_attributes(food_tastes_distances, food_average_distances):
    tastes_weights = {
        "weight": {1: (0, 0.45), 2: (0.45, 0.65), 3: (0.65, 0.75), 4: (0.75, 1)},
        "sweet": {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.7), 4: (0.7, 1)},
        "acid": {1: (0, 0.4), 2: (0.4, 0.6), 3: (0.6, 0.8), 4: (0.8, 1)},
        "salt": {1: (0, 0.5), 2: (0.5, 0.65), 3: (0.65, 0.8), 4: (0.8, 1)},
        "piquant": {1: (0, 0.4), 2: (0.4, 0.55), 3: (0.55, 0.75), 4: (0.75, 1)},
        "fat": {1: (0, 0.5), 2: (0.5, 0.6), 3: (0.6, 0.75), 4: (0.75, 1)},
        "bitter": {1: (0, 0.5), 2: (0.5, 0.6), 3: (0.6, 0.75), 4: (0.75, 1)},
    }
    food_attributes = dict()
    for taste in core_tastes:
        if taste in ["aroma", "flavor"]:
            continue

        similarity = 1 - spatial.distance.cosine(
            food_average_distances[taste]["closest_vec"], food_tastes_distances["aroma"]
        )  # type: ignore
        # scale the similarity using our minmax scaler
        scaled_similarity = minmax_scaler(
            similarity,
            minval=food_average_distances[taste]["farthest"],
            maxval=food_average_distances[taste]["closest"],
        )
        standardized_similarity = check_in_range(
            tastes_weights[taste], scaled_similarity
        )
        food_attributes[taste] = (scaled_similarity, standardized_similarity)
    return food_attributes


def get_the_closest_embedding(food_to_embeddings_dict: dict, dish_embeddings: dict):
    clean_dish_ingredeints = {}
    ingredient_similarity = []
    for dish_ingredient, ingredient_embedding in dish_embeddings.items():
        if dish_ingredient not in food_to_embeddings_dict:
            continue

        if ingredient_embedding[0] == None:
            attempt = dish_ingredient.split("_")
            dish_ingredient = attempt[0]
            ingredient_embedding = np.average(
                food_to_embeddings_dict[dish_ingredient], axis=0
            )

        similarities = []

        for embedding in food_to_embeddings_dict[dish_ingredient]:
            similarities.append(
                1 - spatial.distance.cosine(embedding, ingredient_embedding)
            )

        arg_max = np.argmax(similarities)
        ingredient_similarity.append(similarities[arg_max])
        clean_dish_ingredeints[dish_ingredient] = food_to_embeddings_dict[
            dish_ingredient
        ][arg_max]

    return clean_dish_ingredeints, ingredient_similarity


def compute_embedding_food_ingredients(
    food_ingredients: list, prediction_model: PredictionModel
) -> dict:
    dish_embeddings = {}
    for ingredient in food_ingredients:
        embedding = prediction_model.compute_embedding_for_ingredient(
            " ".join(food_ingredients), ingredient
        )
        dish_embeddings[ingredient] = embedding

    return dish_embeddings


def get_food_attributes(
    food_ingredients: list,
    food_to_embeddings_dict: dict,
    prediction_model: PredictionModel,
):

    dish_embeddings = compute_embedding_food_ingredients(
        food_ingredients, prediction_model
    )

    food_average_distances = get_food_taste_distances_info()
    dish_embeddings, dish_similarities = get_the_closest_embedding(
        food_to_embeddings_dict, dish_embeddings
    )
    food_tastes_distances = calculate_avg_food_vec(
        food_average_distances=food_average_distances,
        dish_ingredients=dish_embeddings,
    )
    if food_tastes_distances == {}:
        return False, False

    # The aroma embedding is the average of all the ingredients in the food
    food_attributes = calculate_food_attributes(
        food_tastes_distances, food_average_distances
    )
    return food_attributes, food_tastes_distances


def get_wine_pairings(wine_recommendations, wine_df, top_n):
    # see if there are any contrasting suggestions
    try:
        (
            contrasting_wines,
            contrasting_nonaromas,
            contrasting_body,
            contrasting_descriptors,
        ) = retrieve_pairing_type_info(
            wine_recommendations, "contrasting", top_n, wine_df
        )
    except:
        contrasting_wines = []

    try:
        (
            congruent_wines,
            congruent_nonaromas,
            congruent_body,
            congruent_descriptors,
        ) = retrieve_pairing_type_info(
            wine_recommendations, "congruent", top_n, wine_df
        )
    except:
        congruent_wines = []

    if len(contrasting_wines) >= 2 and len(congruent_wines) >= 2:
        wine_names = contrasting_wines[:2] + congruent_wines[:3]
        wine_nonaromas = contrasting_nonaromas[:2] + congruent_nonaromas[:3]
        descriptors = contrasting_descriptors[:2] + congruent_descriptors[:3]
        wine_body = contrasting_body[:2] + congruent_body[:3]
        pairing_types = [
            "Contrasting",
            "Contrasting",
            "Congruent",
            "Congruent",
            "Congruent",
        ]
    elif len(contrasting_wines) >= 2:
        wine_names = contrasting_wines
        wine_nonaromas = contrasting_nonaromas
        wine_body = contrasting_body
        descriptors = contrasting_descriptors
        pairing_types = [
            "Contrasting",
            "Contrasting",
            "Contrasting",
            "Contrasting",
            "Contrasting",
        ]
    else:
        wine_names = congruent_wines
        wine_nonaromas = congruent_nonaromas
        wine_body = congruent_body
        descriptors = congruent_descriptors
        pairing_types = [
            "Congruent",
            "Congruent",
            "Congruent",
            "Congruent",
            "Congruent",
        ]

    return {
        "wine_names": wine_names,
        "wine_nonaromas": wine_nonaromas,
        "wine_body": wine_body,
        "pairing_types": pairing_types,
        "descriptors": descriptors,
    }


def generate_pairing_for_ingredients(
    food_to_embeddings_dict: dict,
    ingredients: list,
    wine_recommendations: pd.DataFrame,
    prediction_model: PredictionModel,
):
    food_attributes, food_tastes_distances = get_food_attributes(
        ingredients, food_to_embeddings_dict, prediction_model
    )

    if not food_attributes and not food_tastes_distances:
        return False, False

    wine_recommendations = nonaroma_rules(wine_recommendations, food_attributes)
    wine_recommendations = congruent_or_contrasting(
        wine_recommendations, food_attributes
    )
    wine_recommendations = sort_by_aroma_similarity(
        wine_recommendations, food_tastes_distances
    )
    return food_attributes, wine_recommendations


def generate_pairings(food_dataset: pd.DataFrame):
    prediction_model = PredictionModel()
    food_to_embeddings_dict = get_food_embedding_dict()
    wine_recommendations = normalize_production_wines()

    recipe_pairings_path = Path("./app/data/production/recipe_pairings.csv")
    # if recipe_pairings_path.exists():
    #     recipe_pairings = pd.read_csv(recipe_pairings_path, index_col='Unnamed: 0')

    def iter_recipes(row):
        ingredients = eval(row["ingredients_in_instructions"])

        food_attributes, recommendations = generate_pairing_for_ingredients(
            food_to_embeddings_dict=food_to_embeddings_dict,
            ingredients=ingredients,
            wine_recommendations=wine_recommendations,
            prediction_model=prediction_model,
        )

        if type(recommendations) == bool:
            return [np.nan, np.nan, np.nan, np.nan]

        # ingredients_sample = random.sample(ingredients, k=6)
        # only interested in top 4 wines
        recommended = recommendations.index.T.tolist()[:4]
        return recommended

    food_dataset[
        [
            "first_pairing",
            "second_pairing",
            "third_pairing",
            "fourth_pairing",
        ]
    ] = food_dataset.apply(iter_recipes, axis=1, result_type="expand")

    food_dataset.dropna(inplace=True, axis=0)
    food_dataset.to_csv(recipe_pairings_path, index_label="index")


def main():
    food_dataset = get_food_dataframe()
    food_dataset["ingredients_in_instructions"] = food_dataset[
        "ingredients_in_instructions"
    ].apply(lambda x: (np.nan if x == "[]" else x))
    food_dataset = food_dataset.dropna(subset=["ingredients_in_instructions"])
    food_dataset = food_dataset.sample(n=5000, axis=0, random_state=53)
    generate_pairings(food_dataset=food_dataset)

    # ingredients = ["roast_chicken", "tarragon", "sage"]
    # ingredients = ["pizza_dough", "tomato_sauce", "pepperoni", "mozzarella"]
    # ingredients = [
    #     "oregano",
    #     "onion",
    #     "basil",
    #     "beef",
    #     "green_bean",
    #     "garlic",
    #     "corn",
    #     "chuck",
    #     "carrot",
    #     "potato",
    # ]
    # ingredients = ["smoked_salmon", "dill", "cucumber", "sour_cream"]
    # ingredients = [
    #     "butter",
    #     "ground_black_pepper",
    #     "beef_steak",
    #     "mushroom",
    #     "heavy_cream",
    # ]
    # ingredients = ["beef_roast", "mushroom_gravy", "black_pepper"]
    # ingredients = ["sausage", "tomato", "onion", "pickle", "relish", "mustard"]

    # prediction_model = PredictionModel()
    # food_to_embeddings_dict = get_food_embedding_dict()
    # wine_recommendations = normalize_production_wines()
    # wine_df = get_production_wines()

    # food_attributes, wine_recommendations = generate_pairing_for_ingredients(
    #     food_to_embeddings_dict=food_to_embeddings_dict,
    #     ingredients=ingredients,
    #     wine_recommendations=wine_recommendations,
    #     prediction_model=prediction_model,
    # )

    # wine_pairings = get_wine_pairings(
    #     wine_recommendations,
    #     wine_df,
    #     top_n=4,
    # )

    # plot_wine_recommendations(
    #     ingredients=ingredients,
    #     pairing_wines=wine_pairings["wine_names"],
    #     wine_attributes=wine_pairings["wine_nonaromas"],
    #     pairing_body=wine_pairings["wine_body"],
    #     impactful_descriptors=wine_pairings["descriptors"],
    #     pairing_types=wine_pairings["pairing_types"],
    #     top_n=4,
    #     food_attributes=food_attributes,
    # )


if __name__ == "__main__":
    main()
