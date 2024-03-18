import os
import pandas as pd
import numpy as np
import spacy


from matplotlib import pyplot as plt
from tqdm import tqdm
from notebooks.helpers.prep.normalization import RecipeNormalizer
from notebooks.helpers.prep.frequencies import FrequencyExtractor
from notebooks.helpers.prep.ingredients_mapping import ingredients_mappings
from notebooks.helpers.prep.utils import read_and_write_ingredients

BASE_PATH = "./app/data"


def read_data_and_parse_columns():
    df = pd.read_csv(BASE_PATH + "/food_reviews/RAW_recipes.csv")

    # df[
    #     [
    #         "calories",
    #         "total fat (PDV)",
    #         "sugar (PDV)",
    #         "sodium (PDV)",
    #         "protein (PDV)",
    #         "saturated fat (PDV)",
    #         "carbohydrates (PDV)",
    #     ]
    # ] = df.nutrition.str.split(",", expand=True)
    # df["calories"] = df["calories"].apply(lambda x: x.replace("[", ""))
    # df["carbohydrates (PDV)"] = df["carbohydrates (PDV)"].apply(
    #     lambda x: x.replace("]", "")
    # )
    # df[
    #     [
    #         "calories",
    #         "total fat (PDV)",
    #         "sugar (PDV)",
    #         "sodium (PDV)",
    #         "protein (PDV)",
    #         "saturated fat (PDV)",
    #         "carbohydrates (PDV)",
    #     ]
    # ] = df[
    #     [
    #         "calories",
    #         "total fat (PDV)",
    #         "sugar (PDV)",
    #         "sodium (PDV)",
    #         "protein (PDV)",
    #         "saturated fat (PDV)",
    #         "carbohydrates (PDV)",
    #     ]
    # ].astype(
    #     "float"
    # )
    df.drop(
        [
            "minutes",
            "contributor_id",
            "submitted",
            "tags",
            "nutrition",
            "n_steps",
            "n_ingredients",
            "id",
        ],
        inplace=True,
        axis=1,
    )
    return df


def extract_ingredients(all_raw_ingredients, force=False):
    if ingredients_mappings and not force:
        return ingredients_mappings

    list_ingredients = []
    for ingredients in all_raw_ingredients:
        for ingredient in eval(ingredients):
            # if " and " in ingredient or " or " in ingredient:
            #     ingredient = ingredient.replace(" and ", " ").split(" ")
            #     for ingre in ingredient:
            #         list_ingredients.append(ingre)
            # else:
            list_ingredients.append(ingredient)

    list_ingredients = list(dict.fromkeys(list_ingredients))
    ingredient_normalizer = RecipeNormalizer(lemmatization_types=["NOUN"])

    cleaned_ingredients = ingredient_normalizer.normalize_ingredients(list_ingredients)
    return cleaned_ingredients


def normalize_instructions(instructions_list):
    normalized_instructions = []
    ingredients_in_instructions = []
    instruction_normalizer = RecipeNormalizer()
    for instructions in tqdm(instructions_list, total=len(instructions_list)):
        if instructions is np.nan:
            normalized_instructions.append(None)
            continue

        if type(eval(instructions)) == str:
            instruction_text = [instructions]
        else:
            instruction_text = ["".join(step.strip() for step in eval(instructions))]

        (
            normalized_instruction,
            ingredients_in_instruction,
        ) = instruction_normalizer.normalize_instruction(instruction_text)

        normalized_instructions.append(normalized_instruction)
        ingredients_in_instructions.append(ingredients_in_instruction)

    return normalized_instructions, ingredients_in_instructions


def main():
    ready_ingredients = False
    force = True

    food_dataset = read_data_and_parse_columns()
    food_dataset.dropna(subset=["steps"], inplace=True)

    print(food_dataset.shape)
    food_dataset = food_dataset.iloc[:150000]

    clean_ingredients = extract_ingredients(
        food_dataset.ingredients.to_numpy(), force=force
    )

    if not ready_ingredients:
        read_and_write_ingredients(clean_ingredients)

    if ready_ingredients:

        normalized_instructions_token, ingredients_in_instructions = (
            normalize_instructions(food_dataset["steps"].to_numpy())
        )

        # f_extractor = FrequencyExtractor(
        #     clean_sentences=normalized_instructions_token,
        #     clean_ingredients=clean_ingredients,
        #     type="food",
        # )
        # f_extractor.count_all_ingredients(exclude_rare=True, min_threshold=20)

        # normalized_name_token, _ = normalize_instructions(food_dataset["name"].to_numpy())
        # normalized_description_token, _ = normalize_instructions(
        #     food_dataset["description"].to_numpy()
        # )

        # food_dataset.drop(["steps", "description"], inplace=True, axis=1)

        food_dataset["ingredients_in_instructions"] = ingredients_in_instructions
        food_dataset["clean_instructions"] = normalized_instructions_token
        # food_dataset["clean_description"] = normalized_description_token
        # food_dataset["clean_name"] = normalized_name_token

        food_dataset.to_csv(
            "./app/data/test/reduced_food.csv",
            index_label=False,
            mode="w",
            header=True,
        )


if __name__ == "__main__":
    main()
