import torch
from torch.utils.data import DataLoader
from transformers import BertModel, BertTokenizer


from collections import defaultdict
from pathlib import Path
from .dataset_generator import InstructionsDataset
from ..prep.utils import get_all_ingredients

import numpy as np
import torch


class PredictionModel:
    def __init__(self):
        self.model = BertModel.from_pretrained(
            pretrained_model_name_or_path="./app/notebooks/helpers/models/checkpoint-final"
        )

        self.tokenizer = BertTokenizer(
            vocab_file="./app/notebooks/helpers/models/config/bert-base-cased-vocab.txt",
            do_lower_case=False,
            max_len=256,
            never_split=get_all_ingredients(as_list=True),
        )
        self.device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)

    def predict_embeddings(self, sentences):
        dataset = InstructionsDataset(tokenizer=self.tokenizer, sentences=sentences)
        dataloader = DataLoader(dataset, batch_size=100, pin_memory=True)

        embeddings = []
        ingredient_ids = []
        for batch in dataloader:
            batch = batch.to(self.device)
            with torch.no_grad():
                embeddings_batch = self.model(batch)
                embeddings.extend(embeddings_batch[0])
                ingredient_ids.extend(batch)

        return torch.stack(embeddings), ingredient_ids

    def compute_embedding_for_ingredient(self, sentence, ingredient_name):
        embeddings, ingredient_ids = self.predict_embeddings([sentence])
        embeddings_flat = embeddings.view((-1, 768))
        ingredient_ids_flat = torch.stack(ingredient_ids).flatten()
        food_id = self.tokenizer.convert_tokens_to_ids(ingredient_name)
        food_embedding = embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()

        return food_embedding[0]


def _merge_synonmys(food_to_embeddings_dict, max_sentence_count):
    synonmy_replacements_path = Path(
        "foodbert_embeddings/data/synonmy_replacements.json"
    )
    if synonmy_replacements_path.exists():
        with synonmy_replacements_path.open() as f:
            synonmy_replacements = json.load(f)
    else:
        synonmy_replacements = {}

    merged_dict = defaultdict(list)
    # Merge ingredients
    for key, value in food_to_embeddings_dict.items():
        if key in synonmy_replacements:
            key_to_use = synonmy_replacements[key]
        else:
            key_to_use = key

        merged_dict[key_to_use].append(value)

    merged_dict = {k: np.concatenate(v) for k, v in merged_dict.items()}
    # When embedding count exceeds maximum allowed, reduce back to requested count
    for key, value in merged_dict.items():
        if len(value) > max_sentence_count:
            index = np.random.choice(value.shape[0], max_sentence_count, replace=False)
            new_value = value[index]
            merged_dict[key] = new_value

    return merged_dict
