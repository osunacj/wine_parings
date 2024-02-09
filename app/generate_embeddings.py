import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
import random
import torch
from tqdm import tqdm
import pickle

from notebooks.helpers.models.embedding_model import PredictionModel
from notebooks.helpers.prep.utils import get_all_ingredients
from notebooks.helpers.prep.wine_mapping_values import wine_terms_mappings

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


def get_wine_and_ingredients_reviews():
    wine_dataset = pd.read_csv("./app/data/test/reduced_wines.csv")
    wine_reviews = wine_dataset.loc[:, "clean_descriptions"].to_numpy()[:50]
    ingredients_in_reviews = wine_dataset.loc[:, "descriptors_in_reviews"].to_numpy()[
        :50
    ]
    return wine_reviews, ingredients_in_reviews


def _random_sample_with_min_count(population, k):
    if len(population) <= k:
        return population
    else:
        return random.sample(population, k)


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
    ingredient_ids = model.tokenizer.convert_tokens_to_ids(all_ingredients)
    ingredient_ids_dict = dict(zip(all_ingredients, ingredient_ids))

    return ingredient_ids_dict


def _generate_food_sentence_dict():
    wine_reviews, ingredients_in_reviews = get_wine_and_ingredients_reviews()

    # add food reviews here
    instruction_sentences = wine_reviews
    ingredients_in_sentences = ingredients_in_reviews

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


def generate_food_embedding_dict(max_sentence_count=50):
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )
    if food_to_embeddings_dict_path.exists():
        with food_to_embeddings_dict_path.open("rb") as f:
            food_to_embeddings_dict = pickle.load(f)
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

        with food_to_embeddings_dict_path.open("wb") as f:
            pickle.dump(food_to_embeddings_dict, f)

    return food_to_embeddings_dict


def get_taste_of_ingredient(ingredient):
    ingredient = ingredient.replace("_", " ")
    taste = "flavor"

    if ingredient in wine_terms_mappings:
        taste = wine_terms_mappings[ingredient][2]

    return taste


def construct_taste_ingredient_embeddings(food_to_embeddings_dict):
    wine_dataset = pd.read_csv("./app/data/test/reduced_wines.csv")
    descriptors_in_reviews = wine_dataset.loc[:, "descriptors_in_reviews"].to_numpy()

    constructor = {}
    for core_taste in core_tastes:
        constructor[core_taste + "_descriptors"] = []
        constructor[core_taste + "_embeddings"] = []

    for review in descriptors_in_reviews:
        for core_taste in core_tastes:
            embeddings = []
            ingredients = []
            for ingredient in eval(review):
                taste = get_taste_of_ingredient(ingredient)
                if taste == core_taste:
                    ingredients.append(ingredient)
                    embeddings.append(food_to_embeddings_dict.get(ingredient))
            constructor[core_taste + "_descriptors"].append(ingredients)
            constructor[core_taste + "_embeddings"].append(embeddings)

    embeddings_dataset = pd.DataFrame(constructor, columns=list(constructor.keys()))
    wine_df = pd.concat([wine_dataset, embeddings_dataset], axis=1)


def main():
    food_to_embeddings_dict = generate_food_embedding_dict()
    construct_taste_ingredient_embeddings(food_to_embeddings_dict)


if __name__ == "__main__":
    main()
