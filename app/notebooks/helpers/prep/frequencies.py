import pandas as pd
import numpy as np
from tqdm import tqdm
from .utils import read_and_write_ingredients
import re
from nltk.corpus import stopwords


stop_words = set(stopwords.words("english"))
not_words = [".", ",", "!", "?", " ", ";", ":", "-"]


class FrequencyExtractor:
    def __init__(self, clean_ingredients: dict, clean_sentences: list, type="food"):
        self.clean_ingredients = clean_ingredients
        self.clean_sentences = clean_sentences
        self.type = type

    def count_all_ingredients(
        self,
        min_threshold=50,
        max_threshold=500,
        exclude_rare=True,
    ):
        ingredients_count = {
            key: [ingredient, 0] for key, ingredient in self.clean_ingredients.items()
        }

        for sentence in self.clean_sentences:
            for word in re.findall(r"[\w'-]+|[.,!?;]", sentence):
                if word in stop_words or word in not_words:
                    continue

                if word.replace("_", " ") in ingredients_count:
                    ingredients_count[word.replace("_", " ")][1] += 1

        ingredients_count_sorted = sorted(
            ingredients_count.items(), key=lambda x: x[1][1], reverse=True
        )

        if exclude_rare:
            self.exclude_rare_ingredients(ingredients_count_sorted, min_threshold)

        with open(
            f"./app/notebooks/helpers/temp/{self.type}_ingredients_frequencies.py", "w"
        ) as file:
            file.write(f"frequencies= " + str(ingredients_count_sorted))
            file.close()

        first_threshold = min_threshold + 100
        second_threshold = min_threshold + 200
        third_threshold = max_threshold - 150

        print(
            f"In total found below {min_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1][1] < min_threshold])} ingredients"
        )
        print(
            f"In total found from {min_threshold} to {first_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1][1] >= min_threshold and elem[1][1] < first_threshold])} ingredients"
        )
        print(
            f"In total found from {first_threshold} to {second_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1][1] >= first_threshold and elem[1][1] < second_threshold])} ingredients"
        )
        print(
            f"In total found from {third_threshold} to {max_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1][1] > third_threshold and elem[1][1] < max_threshold])} ingredients"
        )
        print(
            f"In total found above  {max_threshold}: {len([elem for elem in ingredients_count_sorted if elem[1][1] > max_threshold])} ingredients"
        )

        return ingredients_count_sorted

    def exclude_rare_ingredients(self, ingredients_count: list, threshold):
        to_keep_ingredients = {
            key: ingredient[0]
            for key, ingredient in ingredients_count
            if ingredient[1] > threshold
        }

        if self.type == "wine":
            read_and_write_ingredients(
                to_keep_ingredients,
                "./app/notebooks/helpers/prep/wine_descriptors_mapping.py",
                False,
                "wine_descriptors_mapping",
            )
        else:
            read_and_write_ingredients(
                to_keep_ingredients,
            )
