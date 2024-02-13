"""
Used to generate cleaned_recipe1m.json from recipe1m.json by cleaning the instructions
This includes lemmatization, merging multi word ingredients with underscore etc.
"""

from tqdm import tqdm
from typing import List
import spacy
from spacy.tokens import Token
from spacy.language import Language
import re
from nltk.corpus import stopwords
import pandas as pd
import numpy as np

from .ingredients_mapping import ingredients_mappings

stop_words = set(stopwords.words("english"))


def has_number(token):
    # check if string has any numbers
    return any(char.isdigit() for char in token.text)


@Language.component("custom_removel_component")
def custom_removel_component(doc):
    in_paranthesis = False
    stop_words = set(stopwords.words("english"))
    words_to_remove = [
        "/",
        "-",
        "ounce",
        "cup",
        "teaspoon",
        "tbsp",
        "tsp",
        "tablespoon",
        "sm",
        "c",
        "cube",
        "tbsp.",
        "sm.",
        "c.",
        "oz",
        "qt",
        "free",
        "drink",
        "top",
        "mix",
    ]

    for token in doc:
        if token.text == "(":
            in_paranthesis = True

        if (
            not in_paranthesis
            and not token.is_digit
            and token.text not in words_to_remove
            and token.lemma_ not in words_to_remove
            and not has_number(token)
            and token.text[0] != "-"
            and token.lemma_ not in stop_words
        ):
            token._.to_keep = True

        if token.text == ")":
            in_paranthesis = False

    return doc


class RecipeNormalizer:
    """
    Applies both typical normalization and also cleanup according to hardcoded words
    Takes a list of ingredients, uses spacy tokenizer to create docs out of them, applies a .pipe for efficient processing.
    Then creates new ingredients from not eliminated tokens

    Usage Example:
    ingredient_normalizer = IngredientNormalizer()
    ingredient_normalizer.normalize_ingredients(['10 flour tortillas  hot'])
    out: flour tortilla hot

    Originally code from FoodBert modified to custom needs.
    """

    not_word_tokens = [".", ",", "!", "?", " ", ";", ":", "-"]
    words_to_remove = [
        "/",
        "-",
        "ounce",
        "cup",
        "teaspoon",
        "tbsp",
        "tsp",
        "tablespoon",
        "sm",
        "c",
        "cube",
        "tbsp.",
        "sm.",
        "c.",
        "oz",
        "qt",
        "free",
        "drink",
        "top",
        "mix",
    ]

    def __init__(self, lemmatization_types=None, mapping: dict = {}):
        self.model = spacy.load(
            "en_core_web_md", disable=["parser", "ner"]
        )  # Disable parts of the pipeline that are not necessary
        Token.set_extension(
            "to_keep", default=False, force=True
        )  # new attribute for Token
        self.model.add_pipe(
            "custom_removel_component"
        )  # new component for the pipeline
        self.tag_mapping = {
            "NN": "NOUN",
            "NNS": "NOUN",
            "NNP": "NOUN",
            "NNPS": "NOUN",
            ".": "NOUN",
            "JJ": "ADJ",
            "JJS": "ADJ",
            "JJR": "ADJ",
            "VBD": "VERB",
            "VBG": "VERB",
            "VBN": "VERB",
            "VBZ": "VERB",
            "VBP": "VERB",
        }

        # if not None, only lemmatize types in this list
        self.lemmatization_types = lemmatization_types
        if mapping:
            self.mapping = {**mapping, **ingredients_mappings}
        else:
            self.mapping = ingredients_mappings

    def lemmatize_token_to_str(self, token, token_tag):
        if self.lemmatization_types is None or token_tag in self.lemmatization_types:
            lemmatized = token.lemma_.strip()
        else:
            lemmatized = token.text.lower().strip()

        return lemmatized

    def normalize_ingredients(
        self, ingredients: List[str], strict=True, disable=False
    ) -> dict:
        ingredients = [
            ingredient.split(",")[0] for ingredient in ingredients
        ]  # Ignore after comma
        # Disable unnecessary parts of the pipeline, also run at once with pipe which is more efficient
        ingredients_docs = self.model.pipe(ingredients, n_process=-1, batch_size=3000)

        cleaned_ingredients = {}
        for ingredient_doc in tqdm(
            ingredients_docs, total=len(ingredients), disable=disable
        ):
            cleaned_ingredient = []
            for token in ingredient_doc:
                token_tag = token.tag_

                if token_tag in self.tag_mapping:
                    token_tag = self.tag_mapping[token_tag]
                if strict:
                    if len(token) > 1 and token._.to_keep:
                        lemmatized = self.lemmatize_token_to_str(token, token_tag)
                        cleaned_ingredient.append(lemmatized)
                else:
                    if len(token) > 1:
                        lemmatized = self.lemmatize_token_to_str(token, token_tag)
                        cleaned_ingredient.append(lemmatized)

            clean_ingredient = " ".join(cleaned_ingredient)
            cleaned_ingredients.update({clean_ingredient: clean_ingredient})

        return cleaned_ingredients

    def match_ingredients(self, normalized_instruction_tokens, n):
        for i in range(len(normalized_instruction_tokens) - n, -1, -1):
            sublist = normalized_instruction_tokens[i : i + n]
            if (
                sublist[0] in self.not_word_tokens
                or sublist[-1] in self.not_word_tokens
                or sublist[0] in stop_words
                or sublist[-1] in stop_words
            ):
                continue

            clean_sublist = " ".join(
                [token for token in sublist if token not in self.not_word_tokens]
            )

            if clean_sublist in self.mapping:
                new_instruction_tokens = []
                new_ingredient = self.mapping[clean_sublist].replace(" ", "_")
                for idx, token in enumerate(normalized_instruction_tokens):
                    if idx < i or idx >= i + n:
                        new_instruction_tokens.append(token)

                    elif new_ingredient is not None:
                        new_instruction_tokens.append(new_ingredient)
                        new_ingredient = None

                return new_instruction_tokens, True
        return normalized_instruction_tokens, False

    def normalize_instruction(self, instruction):
        instruction_docs = self.model.pipe(instruction, n_process=-1, batch_size=1)
        normalized_instruction = []
        for instruction_doc in instruction_docs:
            for word in instruction_doc:
                if word.text in [" ", "  ", "-"]:
                    continue

                space = " "
                if word.is_punct:  # we want a space before all non-punctuation words
                    space = ""

                if word.tag_ in ["NN", "NNS", "NNP", "NOUN", "NNPS"]:
                    normalized_instruction.append(
                        space
                        + self.lemmatize_token_to_str(token=word, token_tag="NOUN")
                    )
                else:
                    normalized_instruction.append(space + word.text)

        normalized_instruction = "".join(normalized_instruction).strip()

        normalized_instruction_tokens = re.findall(
            r"[\w'-]+|[.,!?; ]", normalized_instruction
        )
        # find all sublists of tokens with descending length
        for n in range(
            5, 1, -1
        ):  # stop at 2 because matching tokens with length 1 can stay as they are
            match = True
            while match:
                normalized_instruction_tokens, match = self.match_ingredients(
                    normalized_instruction_tokens, n
                )

        ingredients_in_instruction = self.extract_ingredients_for_instruction(
            normalized_instruction_tokens
        )
        return (
            "".join(normalized_instruction_tokens).strip(),
            ingredients_in_instruction,
        )

    def extract_ingredients_for_instruction(self, normalized_instruction_tokens):
        ingredients_in_instruction = []
        for token in normalized_instruction_tokens:
            if (
                token in self.not_word_tokens
                or token in stop_words
                or token in self.words_to_remove
            ):
                continue

            if (
                token not in ingredients_in_instruction
                and len(token) > 0
                and token.replace("_", " ") in self.mapping
            ):
                ingredients_in_instruction.append(
                    self.mapping[token.replace("_", " ")].replace(" ", "_")
                )

        return ingredients_in_instruction
