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
        if food_id == 100:
            return [None]
        food_embedding = embeddings_flat[ingredient_ids_flat == food_id].cpu().numpy()
        if len(food_embedding) > 0:
            return food_embedding[0]
