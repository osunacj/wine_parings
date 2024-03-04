import numpy as np
from pathlib import Path

# import pandas as pd
# from scipy import spatial
from sklearn.decomposition import PCA

# import random
# import torch
# from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt
from notebooks.helpers.prep.ingredients_mapping import ingredients_mappings


def get_food_embedding_dict():
    food_to_embeddings_dict_path = Path(
        f"./app/notebooks/helpers/models/food_embeddings_dict_foodbert.pkl"
    )

    with food_to_embeddings_dict_path.open("rb") as f:
        food_to_embeddings_dict = pickle.load(f)

    return food_to_embeddings_dict


def get_food_taste_distances_info():
    food_similarity_dict_path = Path(
        f"./app/notebooks/helpers/models/food_similarity_dict.pkl"
    )

    with food_similarity_dict_path.open("rb") as f:
        food_average_distances = pickle.load(f)
    return food_average_distances


def reduce_embedding_dimensions(food_embeddings, n_components):
    pca = PCA(n_components)
    pca_matrix = pca.fit_transform(food_embeddings)
    return pca_matrix


def make_scatter_with_line(plot, label, x, y):
    plot.scatter(
        x,
        y,
        s=60,
        label=label,
        cmap="plasma",
    )
    # plot.plot(
    #     [0, x],
    #     [0, y],
    # )
    return plot


def plot_pca_vectors_3d(pca_matrix_dict):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")
    for taste, pca_component in pca_matrix_dict.items():
        ax.scatter(
            pca_component[0],
            pca_component[1],
            pca_component[2],
            s=20,
            label=taste,
            marker="o" if "average" in taste else "x",
        )
        ax.text(
            pca_component[0],
            pca_component[1],
            pca_component[2],
            taste,
        )

    ax.set_xlabel("1st Component")
    ax.set_ylabel("2nd Component")
    ax.set_zlabel("3rd Component")
    ax.view_init(elev=14.0, azim=-52, roll=0)
    plt.savefig("./taste_plot_3d.png")
    plt.show()


def plot_pca_vectors_2d(pca_matrix_dict):
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot()
    for taste, pca_component in pca_matrix_dict.items():
        ax.scatter(
            pca_component[0],
            pca_component[1],
            s=20,
            label=taste,
            marker="o" if "average" in taste else "x",
        )
        ax.text(
            pca_component[0],
            pca_component[1],
            taste,
        )

    ax.set_xlabel("1st Component")
    ax.set_ylabel("2nd Component")
    plt.savefig("./taste_plot_2d.png")
    plt.show()


def merge_average_with_ingredients(
    food_to_embedding_dict: dict, food_average_distances, type
):

    merged_embeddings = {}

    if type == "food":
        for ingredient in ingredients_mappings.values():
            ingredient = ingredient.replace(" ", "_")
            if ingredient not in food_to_embedding_dict:
                continue

            embedding = food_to_embedding_dict.get(ingredient)
            embedding = np.average(embedding, axis=0)  # type: ignore
            merged_embeddings[ingredient] = embedding

    for core_taste, embedding in food_average_distances.items():
        merged_embeddings[f"average_{core_taste}"] = embedding["average_vec"]

    return merged_embeddings


def reduce_ingredients_dimension(
    ingredients_embeddings: dict,
    target_ingredients,
    N,
):

    to_reduce = np.stack(list(ingredients_embeddings.values()))

    ingredients_pca = reduce_embedding_dimensions(to_reduce, N)

    for ingredient, ingredient_pca in zip(
        ingredients_embeddings.keys(), ingredients_pca
    ):
        ingredients_embeddings[ingredient] = ingredient_pca

    ingredient_target = {}
    for ingredient in target_ingredients:
        ingredient_target[ingredient] = ingredients_embeddings.get(ingredient)

    return ingredient_target


def main(plot_ingredients):
    N = 2
    food_average_distances = get_food_taste_distances_info()
    get_food_to_embedding_dict = get_food_embedding_dict()

    merged_embeddings = merge_average_with_ingredients(
        get_food_to_embedding_dict, food_average_distances, "food"
    )

    reduced_ingredients_dimension = reduce_ingredients_dimension(
        merged_embeddings, target_ingredients=plot_ingredients, N=N
    )

    if N == 2:
        plot_pca_vectors_2d(reduced_ingredients_dimension)
    else:
        plot_pca_vectors_3d(reduced_ingredients_dimension)


if __name__ == "__main__":
    # plot_ingredients = [
    #     "average_sweet",
    #     "average_salt",
    #     "average_acid",
    #     "average_piquant",
    #     "average_fat",
    #     "average_weight",
    #     "average_bitter",
    # ]

    # plot_ingredients = [
    #     "salty",
    #     "maple_syrup",
    #     "pepper",
    #     "lemon_juice",
    #     "kale",
    #     "fat",
    #     "bacon_grease",
    #     "average_fat",
    #     "butter",
    # ]

    # plot_ingredients = [
    #     "water",
    #     "tap_water",
    #     "boiling_water",
    #     "ice_water",
    #     "warm_water",
    #     "hot_water",
    #     "cold_water",
    # ]

    # plot_ingredients = ["bacon", "salt", "average_salt", "salty", "honey", "raspberry"]

    # plot_ingredients = [
    #     "chicken",
    #     "cooked_chicken",
    #     "chicken_soup",
    #     "chicken_strip",
    #     "chicken_thigh",
    #     "chicken_wing",
    #     "grilled_chicken",
    #     "fried_chicken",
    #     "fish",
    #     "fish_fillet",
    #     "tuna",
    #     "salmon",
    #     "fish_steak",
    # ]
    main(plot_ingredients)
