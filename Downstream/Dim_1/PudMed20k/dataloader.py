import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data

from tqdm import tqdm
from transformers import BertTokenizer

"""
come from https://www.kaggle.com/code/matthewjansen/nlp-medical-abstract-segmentation
"""
class PudMed_20k_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, max_words=112):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.transform = transform
        self.data_path = data_path

        self.classes = {'BACKGROUND': 0, 'CONCLUSIONS': 1, 'METHODS': 2, 'OBJECTIVE': 3, 'RESULTS': 4}
        self.train_samples = self.prepare_raw_data(os.path.join(data_path, split+".txt"))

        self.tokenizer = BertTokenizer.from_pretrained(
            "../Bio_ClinicalBERT/")
        self.max_words = max_words

        print(split, "dataset samples", self.__len__())

    def prepare_raw_data(self, filepath):
        with open(filepath) as f:
            raw_data = f.readlines()

        abstract_data = ""
        abstract_samples = []
        abstract_id = 0

        for line in raw_data:
            if line.startswith("###"):
                abstract_id = int(line.replace("###", "").replace("\n", ""))
                abstract_data = ""
            elif line.isspace():
                abstract_data_split = abstract_data.splitlines()
                for abstract_line_number, abstract_line in enumerate(abstract_data_split):
                    line_data = {}
                    target_text_split = abstract_line.split("\t")
                    line_data["abstract_id"] = abstract_id
                    line_data["line_id"] = f'{abstract_id}_{abstract_line_number}_{len(abstract_data_split)}'
                    line_data["abstract_text"] = target_text_split[1]
                    line_data["line_number"] = abstract_line_number
                    line_data["total_lines"] = len(abstract_data_split)
                    line_data['current_line'] = f'{abstract_line_number}_{len(abstract_data_split)}'
                    line_data["target"] = self.classes[target_text_split[0]]
                    abstract_samples.append(line_data)
            else:
                abstract_data += line
        return abstract_samples

    def __len__(self):
        return len(self.train_samples)

    def get_embedding(self, sent):

        tokens = self.tokenizer(
            sent,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=self.max_words,
        )
        x_len = len([t for t in tokens["input_ids"][0] if t != 0])

        return tokens, x_len

    def __getitem__(self, index):
        items = self.train_samples[index]
        label = torch.Tensor([items["target"]])
        caps, cap_len = self.get_embedding(items["abstract_text"])
        return caps["input_ids"].squeeze(0), label, caps["attention_mask"].squeeze(0)

