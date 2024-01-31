import os
import pandas as pd
import numpy as np
import spacy


from matplotlib import pyplot as plt
from tqdm import tqdm
from notebooks.helpers.prep.foodbert_norm import RecipeNormalizer
from gensim.models.phrases import Phrases, Phraser
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import nltk
import string

# nltk.download('stopwords')

BASE_PATH = "./app/data/wine_extracts"

def merge_all_wine_files():
    i = 0
    wine_dataframe = pd.DataFrame()
    for file in os.listdir(BASE_PATH):
        file_location = BASE_PATH + '/' + str(file)
        if i==0:
            wine_dataframe = pd.read_csv(file_location)
            i+=1
        else:
            df_to_append = pd.read_csv(file_location, low_memory=False, encoding='latin-1')
            wine_dataframe = pd.concat([wine_dataframe, df_to_append], axis=0)

    wine_dataframe.drop_duplicates(subset=['Name'], inplace=True)

    geographies = ['Subregion', 'Region', 'Province', 'Country']

    for geo in geographies:
        wine_dataframe[geo] = wine_dataframe[geo].apply(lambda x : str(x).strip())

    return wine_dataframe

def tokenize_corpus(all_wine_corpus):
    wine_sentences_tokenized = sent_tokenize(all_wine_corpus)
    stop_words = set(stopwords.words('english')) 
    term_normalizer = RecipeNormalizer()

    all_corpus_by_word = []
    for sentence in wine_sentences_tokenized:
        sent = []
        for word in word_tokenize(sentence):
            if word not in stop_words and word.isalpha():
                sent.append(word)
        all_corpus_by_word.append(sent)

    words_in_corpus = [word for sentence in all_corpus_by_word for word in sentence]
    terms = term_normalizer.normalize_ingredients(
        words_in_corpus
    )

    previous_idx = 0
    normalized_corpus_by_words = []
    for sentence in all_corpus_by_word:
        length = len(sentence)
        new_index = previous_idx + length
        normalized_corpus_by_words.append(terms[previous_idx:new_index])
        previous_idx = new_index
   
    return normalized_corpus_by_words
    
def extract_term_frequeuncies_from_bigrams(wine_bigrams):
    wine_bigrams_list = [term for sentence in wine_bigrams for term in sentence]
    wine_terms_count = {term: 0 for term in wine_bigrams_list}

    for term in wine_bigrams_list:
        if term in wine_terms_count:
            wine_terms_count[term] += 1

    wine_terms_sorted = sorted(wine_terms_count.items(), key=lambda x: x[1], reverse=True)

    print(f'In total found from 15 to 100: {len([elem for elem in wine_terms_sorted if elem[1] > 15 and elem[1] < 100])} ingredients')
    print(f'In total found from 100 to 300: {len([elem for elem in wine_terms_sorted if elem[1] > 99 and elem[1] < 300])} ingredients')
    print(f'In total found from 300 to 600: {len([elem for elem in wine_terms_sorted if elem[1] > 299 and elem[1] < 600])} ingredients')

    return wine_terms_sorted

def normalize_wine_reviews(reviews):
    cleaned_ingredients = pd.read_csv('./app/data/food_reviews/ingredients.csv')
    cleaned_ingredients = cleaned_ingredients['0'].to_numpy()

    ingredients_set = {tuple(ing.split(' ')) for ing in cleaned_ingredients}

    normalized_instructions = []
    instruction_normalizer = RecipeNormalizer()
    for instructions in tqdm(reviews, total=len(reviews)):
        if instructions is np.nan:
            normalized_instructions.append(None)
            continue

        if type(instructions) == str:
            instruction_text = [instructions.lower()]
        else:
            instruction_text = [step.strip() for step in eval(instructions)]
        
            
        normalized_instructions.append(
            instruction_normalizer.normalize_instruction(
                instruction_text,
                ingredients_set
            )
        )
    return normalized_instructions


def main():
    wine_dataframe = merge_all_wine_files()

    all_wine_corpus = ' '.join(str(sentence).lower() for sentence in wine_dataframe.Description.to_numpy())

    all_corpus_by_word = tokenize_corpus(all_wine_corpus)

    wine_bigram_model = Phrases(all_corpus_by_word, min_count=10)
    wine_bigrams = [wine_bigram_model[line] for line in all_corpus_by_word]

    wine_terms = extract_term_frequeuncies_from_bigrams(wine_bigrams)

    reviews = wine_dataframe.Description.to_numpy()[:500]
    clean_reviews = normalize_wine_reviews(reviews)


if __name__ == "__main__":
    main()