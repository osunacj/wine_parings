import numpy as np
from pathlib import Path

# import pandas as pd
# from scipy import spatial
from sklearn.decomposition import PCA


from matplotlib import gridspec
from math import pi
import pickle
import matplotlib.pyplot as plt
from .ingredients_mapping import ingredients_mappings


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


def make_spider(grid, n, wine_attributes, title, color, pairing_type, food_attributes):
    # number of variable
    categories = list(food_attributes.keys())
    N = len(categories)

    # What will be the angle of each axis in the plot? (we divide the plot / number of variable)
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(
        grid[n],
        polar=True,
    )

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles[:-1], categories, color="grey", size=11)

    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks(
        [0.25, 0.5, 0.75, 1.0], ["0.25", "0.50", "0.75", "1.00"], color="grey", size=0
    )
    plt.ylim(0, 1)

    # Ind1
    values = list(wine_attributes.values())
    values += values[:1]
    ax.plot(angles, values, color=color, linewidth=2, linestyle="solid")
    ax.fill(angles, values, color=color, alpha=0.4)

    # Add a title
    # Insert a line break in the title if needed
    title_split = str(title).split(",")
    new_title = []
    for number, word in enumerate(title_split):
        if (number % 2) == 0 and number > 0:
            updated_word = "\n" + word.strip()
            new_title.append(updated_word)
        else:
            updated_word = word.strip()
            new_title.append(updated_word)
    new_title = ", ".join(new_title)

    title_incl_pairing_type = new_title + "\n" + "(" + str(pairing_type) + ")"

    plt.title(title_incl_pairing_type, size=13, color="black", y=1.2)


def plot_number_line(gs, n, value, dot_color):
    ax = plt.subplot(gs[n])
    ax.set_xlim(-1, 2)
    ax.set_ylim(0, 3)

    # draw lines
    xmin = 0
    xmax = 1
    y = 1
    height = 0.2

    plt.hlines(y, xmin, xmax)
    plt.vlines(xmin, y - height / 2.0, y + height / 2.0)
    plt.vlines(xmax, y - height / 2.0, y + height / 2.0)

    # draw a point on the line
    px = value
    plt.plot(px, y, "ko", ms=10, mfc=dot_color)

    # add numbers
    plt.text(
        xmin - 0.1,
        y,
        "Light-Bodied",
        horizontalalignment="right",
        fontsize=11,
        color="grey",
    )
    plt.text(
        xmax + 0.1,
        y,
        "Full-Bodied",
        horizontalalignment="left",
        fontsize=11,
        color="grey",
    )

    plt.axis("off")


def plot_wine_recommendations(
    ingredients,
    pairing_wines,
    wine_attributes,
    pairing_body,
    impactful_descriptors,
    pairing_types,
    top_n,
    food_attributes,
):

    plt.figure(figsize=(20, 7), dpi=96)

    grid = gridspec.GridSpec(3, top_n + 1, height_ratios=[3, 0.2, 0.9])

    food_attrtibutes_value = {
        taste: value[0] for taste, value in food_attributes.items()
    }

    length = min(top_n, len(pairing_wines))
    spider_nr = 1
    number_line_nr = spider_nr + length + 1
    descriptor_nr = number_line_nr + length + 1

    plot_food_profile(grid, food_attrtibutes_value, ingredients[:5], n=1 + length)

    for wine in range(length):
        make_spider(
            grid,
            spider_nr,
            wine_attributes[wine],
            pairing_wines[wine],
            "red",
            pairing_types[wine],
            food_attrtibutes_value,
        )
        plot_number_line(grid, number_line_nr, pairing_body[wine], dot_color="red")
        create_text(grid, descriptor_nr, impactful_descriptors[wine])
        spider_nr += 1
        number_line_nr += 1
        descriptor_nr += 1
    plt.show()


def create_text(
    grid, n, impactful_descriptors, text_init="Complementary wine notes: \n\n"
):
    ax = plt.subplot(grid[n])

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.invert_yaxis()

    text = text_init
    text += "\n".join(descriptor for descriptor in impactful_descriptors)

    ax.text(x=0, y=1, s=text, fontsize=12, color="grey")


def plot_food_profile(grid, food_attributes, ingredients, n):
    weight = food_attributes.pop("weight")
    make_spider(
        grid,
        0,
        food_attributes,
        "Food Profile",
        "green",
        "",
        food_attributes,
    )
    plot_number_line(grid, n, weight, dot_color="green")
    create_text(grid, 2 * n, ingredients, text_init="Food ingredients:\n\n")


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
    target_ingredients: list,
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


def view_embeddings_of_ingredient(ingredients: list, targets=None, N=2):
    ## Plots the different embeddings of the same ingredient/s
    get_food_to_embedding_dict = get_food_embedding_dict()
    ingredients_embeddings = {}

    for ingredient in ingredients:
        ingredients_embeddings[ingredient] = get_food_to_embedding_dict.get(ingredient)

    if type(targets) == dict:
        for target_ingredient, target_embedding in targets.items():
            ingredients_embeddings[f"target_{target_ingredient}"] = np.expand_dims(
                target_embedding, axis=0
            )

    embeddings_to_reduce = np.stack(
        np.concatenate([embeddings for embeddings in ingredients_embeddings.values()])
    )

    reduced_embeddings = reduce_embedding_dimensions(embeddings_to_reduce, N)

    count = 0
    for ingredient, embeddings in ingredients_embeddings.items():
        size = len(embeddings)
        ingredients_embeddings[ingredient] = reduced_embeddings[count : count + size]
        count += size

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    for ingredient, pca_components in ingredients_embeddings.items():
        x = []
        y = []
        for pca_component in pca_components:
            x.append(pca_component[0])
            y.append(pca_component[1])

        ax.scatter(
            x,
            y,
            s=20,
            label=ingredient,
            marker="o" if "target" not in ingredient else "x",
        )
    plt.title("PCA of various ingredients")
    plt.legend()
    ax.set_xlabel("1st Component")
    ax.set_ylabel("2nd Component")
    plt.show()


def view_dish_embeddings(
    ingredients=[],
    dish_embedding=[],
    N=2,
    type="food",
    include_avg=["acid", "sweet", "weight", "bitter", "piquant", "fat", "salt"],
):
    if not ingredients and len(dish_embedding) < 1:
        raise Exception

    ## Views the ingredients in a dish
    food_average_distances = get_food_taste_distances_info()
    get_food_to_embedding_dict = get_food_embedding_dict()

    merged_embeddings = merge_average_with_ingredients(
        get_food_to_embedding_dict, food_average_distances, type
    )

    if len(dish_embedding) > 0:
        merged_embeddings["dish"] = dish_embedding
        ingredients.append("dish")

    for avg_taste in include_avg:
        ingredients.append(f"average_{avg_taste}")

    reduced_ingredients_dimension = reduce_ingredients_dimension(
        merged_embeddings, target_ingredients=ingredients, N=N
    )

    if N == 2:
        plot_pca_vectors_2d(reduced_ingredients_dimension)
    else:
        plot_pca_vectors_3d(reduced_ingredients_dimension)


def main():
    view_embeddings_of_ingredient(["chicken", "banana", "meat", "cherry", "oil"], 2)


if __name__ == "__main__":

    main()
