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


def get_wine_and_ingredients_reviews():
    wine_dataset = pd.read_csv("./app/data/test/reduced_wines.csv")
    wine_reviews = wine_dataset.loc[:, "clean_descriptions"].to_numpy()
    ingredients_in_reviews = wine_dataset.loc[:, "descriptors_in_reviews"].to_numpy()
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
    all_ingredients = get_all_ingredients(as_list=True)
    model = PredictionModel()
    ingredient_ids = model.tokenizer.convert_tokens_to_ids(all_ingredients)

    ingredient_ids_dict = dict(zip(all_ingredients, ingredient_ids))

    return ingredient_ids_dict


def _generate_food_sentence_dict():
    wine_reviews, ingredients_in_reviews = get_wine_and_ingredients_reviews()

    food_items = get_all_ingredients(as_list=True)

    # add food reviews here
    instruction_sentences = wine_reviews
    ingredients_in_sentences = ingredients_in_reviews

    food_to_sentences_dict = defaultdict(list)
    for sentence, ingredients_in_sentence in zip(
        instruction_sentences, ingredients_in_sentences
    ):
        for ingredient in eval(ingredients_in_sentence):
            if ingredient in food_items and ingredient in sentence:
                food_to_sentences_dict[ingredient].append(sentence)

    return food_to_sentences_dict


def generate_food_embedding_dict(max_sentence_count=50):
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )
    if food_to_embeddings_dict_path.exists():
        with food_to_embeddings_dict_path.open("rb") as f:
            food_to_embeddings_dict = pickle.load(f)

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
        food_embeddings = embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()
        food_to_embeddings_dict[food].extend(food_embeddings)

    food_to_embeddings_dict = {
        k: np.stack(v) for k, v in food_to_embeddings_dict.items()
    }

    with food_to_embeddings_dict_path.open("wb") as f:
        pickle.dump(food_to_embeddings_dict, f)

    return food_to_embeddings_dict


def main():
    generate_food_embedding_dict()


if __name__ == "__main__":
    main()
