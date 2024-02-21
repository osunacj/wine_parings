from pathlib import Path
from .ingredients_mapping import ingredients_mappings
from .wine_descriptors_mapping import wine_descriptors_mapping
from typing import Union, List, Dict
import matplotlib.pyplot as plt
from matplotlib import gridspec
from math import pi


def get_all_ingredients(as_list=True) -> Union[Dict[str, str], List[str]]:
    all_ingredients = {**wine_descriptors_mapping, **ingredients_mappings}

    if as_list:
        ingredients = []
        for value in all_ingredients.values():
            value = value.replace(" ", "_")
            if value not in ingredients and len(value) > 0:
                ingredients.append(value)
        return sorted(ingredients)

    return all_ingredients


def modify_vocabulary():
    bert_vocab_path = Path(
        "./app/notebooks/helpers/models/config/bert-base-cased-vocab.txt"
    )

    with bert_vocab_path.open() as f:
        bert_vocab = f.read().splitlines()

    ingredients = get_all_ingredients()

    ingredients_to_add = [
        ingredient for ingredient in ingredients if ingredient not in bert_vocab
    ]

    if len(ingredients_to_add) > 0:
        with bert_vocab_path.open(mode="a") as f:
            f.write("\n")
            f.write("\n".join(ingredients_to_add))

    print(f"\nA total of {len(ingredients_to_add)} were added to vocab.")


def read_and_write_ingredients(
    new_ingredients: dict = {},
    custom_path: str = "",
    append_ingredients: bool = True,
    variable_name: str = "ingredients_mappings",
) -> dict:
    if len(custom_path) != 0:
        file_path = custom_path
    else:
        file_path = "./app/notebooks/helpers/prep/ingredients_mapping.py"

    if append_ingredients:
        new_ingredients.update(ingredients_mappings)

    new_ingredients = {
        elem.strip(): value
        for elem, value in new_ingredients.items()
        if len(elem.split()) <= 3 and len(elem) > 1
    }
    new_ingredients_sorted = sorted(new_ingredients.items(), key=lambda item: item[0])

    new_ingredients = {key: value for key, value in new_ingredients_sorted}

    with open(file_path, "w") as file:
        file.write(f"{variable_name} = " + str(new_ingredients))
        file.close()

    return new_ingredients


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
    pairing_wines,
    wine_attributes,
    pairing_body,
    impactful_descriptors,
    pairing_types,
    top_n,
    food_attributes,
):

    plt.figure(figsize=(20, 7), dpi=96)

    grid = gridspec.GridSpec(3, top_n, height_ratios=[3, 0.5, 1])

    spider_nr = 0
    number_line_nr = 4
    descriptor_nr = 8

    for wine in range(min(top_n, len(pairing_wines))):
        make_spider(
            grid,
            spider_nr,
            wine_attributes[wine],
            pairing_wines[wine],
            "red",
            pairing_types[wine],
            food_attributes,
        )
        plot_number_line(grid, number_line_nr, pairing_body[wine], dot_color="red")
        # create_text(gs, descriptor_nr, impactful_descriptors[wine])
        spider_nr += 1
        number_line_nr += 1
        descriptor_nr += 1
    plt.show()


if __name__ == "__main__":
    # ingredients = get_all_ingredients(True)
    modify_vocabulary()
