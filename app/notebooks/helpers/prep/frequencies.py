import pandas as pd
import numpy as np
from tqdm import tqdm




class FrquencyExtractor():

    def __init__(self, clean_ingredients: list, clean_sentences: list ):
        self.clean_ingredients = clean_ingredients
        self.clean_sentences = clean_sentences

    def count_all_ingredients(self):
        ingredients_set = {'_'.join(ingredient.split()) for ingredient in self.clean_ingredients}
        ingredients_count = {ingredient: 0 for ingredient in ingredients_set}

        not_word_tokens = ['.', ',', '!', '?', ';', ':', '-']

        for instruction in tqdm(self.clean_sentences, total=len(self.clean_sentences)):

            for not_word_token in not_word_tokens:
                instruction = instruction.replace(not_word_token, '')

            instruction_words = instruction.split()
            for word in instruction_words:
                if word in ingredients_count:
                    ingredients_count[word] += 1
                
        ingredients_count_sorted = sorted(ingredients_count.items(), key=lambda x: x[1], reverse=True)

        
        print(f'In total found: {len([elem for elem in ingredients_count_sorted if elem[1] > 0])} ingredients')
        print(f'More than 10 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 10])} ingredients')
        print(f'More than 20 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 20])} ingredients')
        print(f'More than 100 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 100])} ingredients')
        print(f'More than 1000 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 1000])} ingredients')
        print(f'More than 10000 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 10000])} ingredients')
        print(f'More than 100000 times: {len([elem for elem in ingredients_count_sorted if elem[1] > 100000])} ingredients')

        return ingredients_count_sorted

    def exlude_rare_ingredients(self, ingredients_count:dict, threshold):
        with open('./app/data/') as f:
            vocab = f.read().splitlines() 
        to_keep_ingredients = [ingredient for ingredient, counts in ingredients_count if counts > threshold]

        ingredients_to_add = []
        for ingredient_to_keep in to_keep_ingredients:
            if ingredient_to_keep not in vocab:
                ingredients_to_add.append(ingredient_to_keep)

        with open('file_to_append',mode='a') as file_append:
            file_append.write('\n'.join(ingredients_to_add))

        print(len(ingredients_to_add))