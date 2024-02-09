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


if __name__ == "__main__":
    # ingredients = get_all_ingredients(True)
    modify_vocabulary()
