from pathlib import Path
from .ingredients_mapping import ingredients_mappings
from .wine_descriptors_mapping import wine_descriptors_mapping
from typing import Union, List, Dict


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


if __name__ == "__main__":
    # ingredients = get_all_ingredients(True)
    modify_vocabulary()
