import pandas as pd
import numpy as np
from tqdm import tqdm


class FrequencyExtractor:
    def __init__(self, clean_ingredients: list, clean_sentences: list):
        self.clean_ingredients = clean_ingredients
        self.clean_sentences = clean_sentences

    def count_all_ingredients(self, min_threshold=50, max_threshold=500):
        ingredients_count = {
            ingredient.replace(" ", "_"): 0 for ingredient in self.clean_ingredients
        }

        for sentence in self.clean_sentences:
            for ingredient in self.clean_ingredients:
                ingredient = ingredient.replace(" ", "_")
                if ingredient in ingredients_count and ingredient in sentence:
                    ingredients_count[ingredient] += 1

        ingredients_count_sorted = sorted(
            ingredients_count.items(), key=lambda x: x[1], reverse=True
        )

        with open(
            "./app/notebooks/helpers/prep/food_ingredients_frequencies.py", "w"
        ) as file:
            file.write(f"frequencies= " + str(ingredients_count_sorted))
            file.close()

        first_threshold = min_threshold + 100
        second_threshold = min_threshold + 200
        third_threshold = max_threshold - 150

        print(
            f"In total found below {min_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1] < min_threshold])} ingredients"
        )
        print(
            f"In total found from {min_threshold} to {first_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1] >= min_threshold and elem[1] < first_threshold])} ingredients"
        )
        print(
            f"In total found from {first_threshold} to {second_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1] >= first_threshold and elem[1] < second_threshold])} ingredients"
        )
        print(
            f"In total found from {third_threshold} to {max_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1] > third_threshold and elem[1] < max_threshold])} ingredients"
        )
        print(
            f"In total found above  {max_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1] > max_threshold])} ingredients"
        )

        return ingredients_count_sorted

    def exlude_rare_ingredients(self, ingredients_count: dict, threshold):
        with open("./app/data/") as f:
            vocab = f.read().splitlines()
        to_keep_ingredients = [
            ingredient for ingredient, counts in ingredients_count if counts > threshold
        ]

        ingredients_to_add = []
        for ingredient_to_keep in to_keep_ingredients:
            if ingredient_to_keep not in vocab:
                ingredients_to_add.append(ingredient_to_keep)

        with open("file_to_append", mode="a") as file_append:
            file_append.write("\n".join(ingredients_to_add))

        print(len(ingredients_to_add))
