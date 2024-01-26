import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import spacy

BASE_PATH = "./app/data/food_reviews"


def merge_interactions_with_recipes():
    raw_recipes = pd.read_csv(BASE_PATH + "/RAW_recipes.csv")
    raw_interactions = pd.read_csv(BASE_PATH + "/RAW_interactions.csv")
    # food_df = raw_recipes.merge(raw_interactions.drop(["rating","date"], axis=1), left_on=["id"], right_on=["recipe_id"], how='left')
    return raw_recipes

def remove_columns(df):
    df[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']] = df.nutrition.str.split(",",expand=True) 
    df['calories'] =  df['calories'].apply(lambda x: x.replace('[','')) 
    df['carbohydrates (PDV)'] =  df['carbohydrates (PDV)'].apply(lambda x: x.replace(']','')) 
    df[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']] = df[['calories','total fat (PDV)','sugar (PDV)','sodium (PDV)','protein (PDV)','saturated fat (PDV)','carbohydrates (PDV)']].astype('float')
    df.drop(['minutes', 'contributor_id', 'submitted', 'tags', 'nutrition', 'n_steps'], inplace=True, axis=1)
    return df





def has_number(token):
    # check if string has any numbers
    return any(char.isdigit() for char in token.text)

def custom_removel_component(doc):
    words_to_remove = ['/', '-', 'ounce', 'cup', 'teaspoon', 'tbsp', 'tsp', 'tablespoon', 'sm', 'c', 'cube', 'tbsp.', 'sm.', 'c.', 'oz']
    in_paranthesis = False
    for token in doc:
        if token.text == '(':
            in_paranthesis = True

        if not in_paranthesis and not token.is_digit and token.text not in words_to_remove and token.lemma_ not in words_to_remove and not has_number(token) and token.text[0] != '-':
            continue

        if token.text == ')':
            in_paranthesis = False

    return " ".join([token.lemma_ for token in doc])

def normalize_instructions(instructions_list):
    model = spacy.load("en_core_web_sm", disable=['parser', 'ner'])
    normalized_instructions = []
    for instructions in instructions_list:
        steps = ", ".join([step for step in instructions])
        step = model(steps)
        step = custom_removel_component(step)
        normalized_instructions.append(step)
    return normalized_instructions




def main():
    food_df = merge_interactions_with_recipes()
    food_df = remove_columns(food_df)
    # normalized_instructions = normalize_instructions(food_df['steps'])
    normalize_instruction = normalize_instructions( [['preheat oven to 325f', 'sprinkle the roast with garlic powder , oregano and pepper , then cook roast , uncovered , in a shallow roasting pan , about 30 minutes per pound', "roast will be very rare-- don't overcook it !", 'let cool slightly , then thinly slice', "add to the roast's pan drippings: the boiling water , bouillon cubes , oregano , thyme , pepper , tabasco sauce , garlic and worcestershire sauce", 'simmer for 20 minutes , scraping up the browned bits', 'taste for salt and add some if you wish', 'add the sliced beef', 'cover and marinate in da gravy overnight', 'reheat the next day and serve in crusty italian sandwich rolls']])



if __name__ == '__main__':
    main()