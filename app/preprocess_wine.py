from distutils.command import clean
import os
import pandas as pd
import numpy as np
import spacy


from matplotlib import pyplot as plt
from tqdm import tqdm
from notebooks.helpers.prep.normalization import RecipeNormalizer
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import word_tokenize, sent_tokenize
from notebooks.helpers.prep.wine_mapping_values import wine_terms_mappings
from notebooks.helpers.prep.wine_descriptors_mapping import wine_descriptors_mapping
from nltk.corpus import stopwords
import nltk
import string

# nltk.download('stopwords')

BASE_PATH = "./app/data/wine_extracts"


def merge_all_wine_files(write=False):
    i = 0
    wine_dataframe = pd.DataFrame()
    for file in os.listdir(BASE_PATH)[:3]:
        file_location = BASE_PATH + "/" + str(file)
        if i == 0:
            wine_dataframe = pd.read_csv(file_location)
            i += 1
        else:
            df_to_append = pd.read_csv(
                file_location, low_memory=False, encoding="latin-1"
            )
            wine_dataframe = pd.concat(
                [wine_dataframe, df_to_append], axis=0, ignore_index=False
            )

    wine_dataframe.drop_duplicates(subset=["Name"], inplace=True)

    geographies = ["Subregion", "Region", "Province", "Country"]

    for geo in geographies:
        wine_dataframe[geo] = wine_dataframe[geo].apply(lambda x: str(x).strip())

    if write:
        wine_dataframe.drop(["Unnamed: 0"], inplace=True, axis=1)
        wine_dataframe.to_csv("./app/data/produce/wine_data.csv", index_label=False)

    return wine_dataframe


def normalize_wine_terms_corpus(all_wine_corpus):
    wine_sentences = sent_tokenize(all_wine_corpus)
    stop_words = set(stopwords.words("english"))

    normalized_corpus_by_sentences = []
    for sentence in tqdm(wine_sentences, total=len(wine_sentences)):
        tokenized_sentence = []
        for word in word_tokenize(sentence):
            if word not in stop_words and word.isalpha():
                tokenized_sentence.append(word)

        normalized_corpus_by_sentences.append(tokenized_sentence)
    return normalized_corpus_by_sentences


def extract_term_frequeuncies_from_bigrams(wine_bigrams, max_threshold, min_threshold):
    wine_bigrams_list = [term for sentence in wine_bigrams for term in sentence]
    wine_terms_count = {term: 0 for term in wine_bigrams_list}

    for term in wine_bigrams_list:
        if term in wine_terms_count:
            wine_terms_count[term] += 1

    wine_terms_sorted = sorted(
        wine_terms_count.items(), key=lambda x: x[1], reverse=True
    )

    with open("./app/notebooks/helpers/prep/wine_term_frequencies.py", "w") as file:
        file.write(f"frequencies= " + str(wine_terms_sorted))
        file.close()

    first_threshold = min_threshold + 100
    second_threshold = min_threshold + 200
    third_threshold = max_threshold - 100

    print(
        f"In total found below {min_threshold}: {len([elem for elem in wine_terms_sorted if elem[1] < min_threshold])} ingredients"
    )
    print(
        f"In total found from {min_threshold} to {first_threshold}: {len([elem for elem in wine_terms_sorted if elem[1] >= min_threshold and elem[1] < first_threshold])} ingredients"
    )
    print(
        f"In total found from {first_threshold} to {second_threshold}: {len([elem for elem in wine_terms_sorted if elem[1] >= first_threshold and elem[1] < second_threshold])} ingredients"
    )
    print(
        f"In total found from {third_threshold} to {max_threshold}: {len([elem for elem in wine_terms_sorted if elem[1] > third_threshold and elem[1] < max_threshold])} ingredients"
    )
    print(
        f"In total found above  {max_threshold}: {len([elem for elem in wine_terms_sorted if elem[1] > max_threshold])} ingredients"
    )

    wine_terms_sorted = [
        elem[0].replace("_", " ")
        for elem in wine_terms_sorted
        if elem[1] >= min_threshold and elem[1] <= max_threshold
    ]

    return wine_terms_sorted


def normalize_wine_descriptors_as_ingredients(wine_descriptors: list = []):
    descriptor_normalizer = RecipeNormalizer(lemmatization_types=["NOUN", "ADJ"])

    if not wine_descriptors and wine_descriptors_mapping:
        return wine_descriptors_mapping

    if not wine_descriptors_mapping:
        normalized_descriptors = descriptor_normalizer.normalize_ingredients(
            wine_descriptors
        )
    else:
        normalized_descriptors = descriptor_normalizer.normalize_ingredients(
            wine_descriptors
        )
        normalized_descriptors.update(wine_descriptors_mapping)

    normalized_descriptors = descriptor_normalizer.read_and_write_ingredients(
        normalized_descriptors,
        "./app/notebooks/helpers/prep/wine_descriptors_mapping.py",
        False,
        "wine_descriptors_mapping",
    )

    return normalized_descriptors


def normalize_wine_reviews(reviews, normalized_descriptors_mapping):
    normalized_instructions = []
    wine_normalizer = RecipeNormalizer(mapping=normalized_descriptors_mapping)
    for instructions in tqdm(reviews, total=len(reviews)):
        if instructions is np.nan:
            normalized_instructions.append(None)
            continue

        if type(instructions) == str:
            instruction_text = [instructions.lower()]
        else:
            instruction_text = [step.strip() for step in eval(instructions)]

        normalized_instructions.append(
            wine_normalizer.normalize_instruction(instruction_text)
        )

    return normalized_instructions


def main():
    # wine_dataframe = merge_all_wine_files(write=True)
    wine_dataframe = pd.read_csv("./app/data/produce/wine_data.csv")

    # all_wine_corpus = " ".join(
    #     str(sentence) for sentence in wine_dataframe.Description.to_numpy()[:10000]
    # ).lower()

    # normalized_corpus_by_sentences = normalize_wine_terms_corpus(all_wine_corpus)

    # wine_bigram_model = Phrases(normalized_corpus_by_sentences, min_count=10)
    # wine_bigrams = [wine_bigram_model[line] for line in normalized_corpus_by_sentences]
    # wine_trigram_model = Phrases(wine_bigrams, min_count=5)

    # # wine_trigram_model.save("wine_trigrams.pkl")
    # # wine_trigram_model = Phraser.load('wine_trigrams.pkl')

    # phrased_wine_sentences = [wine_trigram_model[line] for line in wine_bigrams]

    # descriptors = extract_term_frequeuncies_from_bigrams(
    #     phrased_wine_sentences, max_threshold=500, min_threshold=15
    # )

    # descriptors = [*descriptors, *list(wine_terms_mappings)]
    normalized_descriptors = normalize_wine_descriptors_as_ingredients()

    reviews = wine_dataframe.Description.to_numpy()[:500]
    clean_reviews = normalize_wine_reviews(reviews, normalized_descriptors)
    wine_dataframe["clean_descriptions"] = None
    wine_dataframe["clean_descriptions"].iloc[:500] = clean_reviews


if __name__ == "__main__":
    main()
