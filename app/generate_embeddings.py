import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
from sklearn.decomposition import PCA
import random
import torch
from tqdm import tqdm
import pickle

from notebooks.helpers.models.embedding_model import PredictionModel
from notebooks.helpers.prep.utils import get_all_ingredients, modify_vocabulary
from notebooks.helpers.prep.wine_mapping_values import wine_terms_mappings
from notebooks.helpers.prep.synonmy_replacements import synonyms

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
    wine_reviews = wine_dataset.loc[:, "clean_descriptions"].to_numpy()
    ingredients_in_reviews = wine_dataset.loc[:, "descriptors_in_reviews"].to_numpy()[
        :50
    ]
    return wine_reviews, ingredients_in_reviews


def _random_sample_with_min_count(population, k):
    if len(population) <= k:
        return population
    else:
        return random.sample(population, k)


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


def _merge_synonmys(food_to_embeddings_dict):
    synonmy_replacements_path = Path(
        "./app/notebooks/helpers/prep/synonmy_replacements.py"
    )
    if synonmy_replacements_path.exists() and synonyms:
        synonmy_replacements = synonyms
    else:
        synonmy_replacements = {}

    # Map synonyms mapping
    for key in food_to_embeddings_dict.keys():
        if key in synonmy_replacements:
            key_to_use = synonmy_replacements[key]
            food_to_embeddings_dict[key] = food_to_embeddings_dict[key_to_use]

    return food_to_embeddings_dict


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


def generate_food_embedding_dict(max_sentence_count=50):
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )
    if food_to_embeddings_dict_path.exists():
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


def construct_taste_ingredient_embeddings():
    food_to_embeddings_dict = generate_food_embedding_dict()

    wine_dataset = pd.read_csv("./app/data/test/reduced_wines.csv")
    descriptors_in_reviews = wine_dataset.loc[:, "descriptors_in_reviews"].to_numpy()

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
                avg_embedding = np.nan
                if ingredient in food_to_embeddings_dict:
                    avg_embedding = np.average(
                        food_to_embeddings_dict.get(ingredient), axis=0  # type: ignore
                    )  # type: ignore
                if taste == core_taste:
                    ingredients.append(ingredient)
                    embeddings.append(avg_embedding)
            embeddings = list(filter(lambda x: type(x) is not float, embeddings))
            constructor[core_taste + "_embeddings"].append(
                np.average(embeddings, axis=0) if embeddings else np.nan
            )
            constructor[core_taste + "_descriptors"].append(ingredients)

    embeddings_dataset = pd.DataFrame(constructor, columns=list(constructor.keys()))

    return embeddings_dataset


def average_taste_embeddings(dataframe):
    # pull the average embedding for the wine attribute across all wines.
    avg_taste_embeddings = dict()
    for core_taste in core_tastes:
        # look at the average embedding for a taste, across all wines that have descriptors for that taste
        review_arrays = dataframe[core_taste + "_embeddings"].dropna()
        avg_taste_embeddings[core_taste] = np.average(review_arrays, axis=0)
    return avg_taste_embeddings


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


def wine_varieties(dataframe: pd.DataFrame):
    avg_taste_embeddings = average_taste_embeddings(dataframe)
    wine_varieties = list(set(zip(dataframe["Variety"], dataframe["geo_normalized"])))
    wine_varieties = list(
        filter(lambda x: type(x[0]) == str and type(x[1]) == str, wine_varieties)
    )

    wine_varieties_index = [f"{variety} {geo}" for variety, geo in wine_varieties]

    taste_words_dataframes = []
    taste_variety_dataframes = []

    for taste in core_tastes:

        wine_variety_vectors = []
        wine_variety_top_words = []
        for variety in wine_varieties:
            wines_in_variety = dataframe.loc[
                (dataframe["Variety"] == variety[0])
                & (dataframe["geo_normalized"] == variety[1])
            ]

            a = wines_in_variety.shape[0]
            b = str(variety[1][-1]) == "0"
            if a < 1 or b:
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

            average_variety_vec = np.average(taste_embeddings, axis=0)

            top_n_words = get_top_words_in_variety(wines_in_variety, taste, n=50)

            wine_variety_vectors.append(average_variety_vec)
            wine_variety_top_words.append(top_n_words)

        if taste != "aroma":
            pca = PCA(3)
            wine_variety_vectors = pca.fit_transform(wine_variety_vectors)
            wine_variety_vectors = pd.DataFrame(
                wine_variety_vectors,
                index=wine_varieties_index,
                columns=[f"{taste}_first", f"{taste}_second", f"{taste}_third"],
            )
        else:
            wine_variety_vectors = pd.Series(
                wine_variety_vectors, index=wine_varieties_index, name="embedding"
            )

        wine_variety_vectors.sort_index(inplace=True)
        wine_variety_descriptions = pd.DataFrame(
            wine_variety_top_words, index=wine_varieties_index
        )
        wine_variety_descriptions = pd.melt(
            wine_variety_descriptions.reset_index(), id_vars="index"
        )
        wine_variety_descriptions.sort_index(inplace=True)

        taste_variety_dataframes.append(wine_variety_vectors)

    wines = pd.concat(taste_variety_dataframes, axis=1, ignore_index=False)
    return wines


def main():
    modify_vocabulary()
    wine_dataset = pd.read_csv("./app/data/test/reduced_wines.csv")

    embeddings_dataframe = construct_taste_ingredient_embeddings()

    wine_df = pd.concat([wine_dataset, embeddings_dataframe], axis=1)
    wines_varieties = wine_varieties(dataframe=wine_df)

    wines_varieties.to_csv("./app/data/production/wine_production.csv")


if __name__ == "__main__":
    main()
