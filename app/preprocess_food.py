import os
import pandas as pd
import numpy as np
import spacy


from matplotlib import pyplot as plt
from tqdm import tqdm
from notebooks.helpers.prep.normalization import RecipeNormalizer


BASE_PATH = "./app/data"


def read_data_and_parse_columns():
    df = pd.read_csv(BASE_PATH + "/food_reviews/RAW_recipes.csv")

    df[
        [
            "calories",
            "total fat (PDV)",
            "sugar (PDV)",
            "sodium (PDV)",
            "protein (PDV)",
            "saturated fat (PDV)",
            "carbohydrates (PDV)",
        ]
    ] = df.nutrition.str.split(",", expand=True)
    df["calories"] = df["calories"].apply(lambda x: x.replace("[", ""))
    df["carbohydrates (PDV)"] = df["carbohydrates (PDV)"].apply(
        lambda x: x.replace("]", "")
    )
    df[
        [
            "calories",
            "total fat (PDV)",
            "sugar (PDV)",
            "sodium (PDV)",
            "protein (PDV)",
            "saturated fat (PDV)",
            "carbohydrates (PDV)",
        ]
    ] = df[
        [
            "calories",
            "total fat (PDV)",
            "sugar (PDV)",
            "sodium (PDV)",
            "protein (PDV)",
            "saturated fat (PDV)",
            "carbohydrates (PDV)",
        ]
    ].astype(
        "float"
    )
    df.drop(
        [
            "minutes",
            "contributor_id",
            "submitted",
            "tags",
            "nutrition",
            "n_steps",
            "n_ingredients",
        ],
        inplace=True,
        axis=1,
    )
    return df


def extract_ingredients(all_raw_ingredients):
    list_ingredients = []
    for ingredients in tqdm(all_raw_ingredients, total=len(all_raw_ingredients)):
        for ingredient in eval(ingredients):
            if " and " in ingredient or " or " in ingredient:
                ingredient = ingredient.replace(" and ", " ").split(" ")
                for ingre in ingredient:
                    list_ingredients.append(ingre)
            else:
                list_ingredients.append(ingredient)

    list_ingredients = list(dict.fromkeys(list_ingredients))
    ingredient_normalizer = RecipeNormalizer(lemmatization_types=["NOUN"])

    cleaned_ingredients = ingredient_normalizer.normalize_ingredients(list_ingredients)
    cleaned_ingredients = ingredient_normalizer.read_and_write_ingredients(
        cleaned_ingredients
    )

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
            instruction_text = [step.strip() for step in eval(instructions)]

        (
            normalized_instruction,
            ingredients_in_instruction,
        ) = instruction_normalizer.normalize_instruction(instruction_text)

        normalized_instructions.append(normalized_instruction)
        ingredients_in_instructions.append(ingredients_in_instruction)

    return normalized_instructions, ingredients_in_instructions


def main():
    food_dataset = read_data_and_parse_columns()

    extract_ingredients(food_dataset.ingredients.to_numpy())

    normalized_instructions_token, ingredients_in_instruction = normalize_instructions(
        food_dataset["steps"].to_numpy()[:50]
    )
    normalized_name_token, _ = normalize_instructions(food_dataset["name"].to_numpy())
    normalized_description_token, _ = normalize_instructions(
        food_dataset["description"].to_numpy()[:500]
    )

    food_dataset.drop(["name", "steps", "description"], inplace=True, axis=1)

    food_dataset[["clean_instructions", "clean_description", "clean_name"]] = None

    food_dataset["clean_instructions"][:500] = normalized_instructions_token
    food_dataset["clean_description"][:500] = normalized_description_token
    food_dataset["clean_name"][:500] = normalized_name_token


if __name__ == "__main__":
    main()
