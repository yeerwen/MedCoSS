import os
import pickle
import re
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from nltk.tokenize import RegexpTokenizer
from tqdm import tqdm
from transformers import BertTokenizer
import cv2
import pydicom
from PIL import Image
from transformers import (
    DataCollatorForLanguageModeling,
    DataCollatorForWholeWordMask,
    BertTokenizerFast,
    RobertaTokenizerFast
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# print("base_dir", BASE_DIR)
def read_from_dicom(img_path, imsize=None, transform=None):
    dcm = pydicom.read_file(img_path)
    x = dcm.pixel_array

    x = cv2.convertScaleAbs(x, alpha=(255.0 / x.max()))
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        x = cv2.bitwise_not(x)

    # transform images
    if imsize is not None:
        x = resize_img(x, imsize)

    img = Image.fromarray(x).convert("RGB")

    if transform is not None:
        img = transform(img)

    return img


def resize_img(img, scale):
    """
    Args:
        img - image as numpy array (cv2)
        scale - desired output image-size as scale x scale
    Return:
        image resized to scale x scale with shortest dimension 0-padded
    """
    size = img.shape
    max_dim = max(size)
    max_ind = size.index(max_dim)

    # Resizing
    if max_ind == 0:
        # image is heigher
        wpercent = scale / float(size[0])
        hsize = int((float(size[1]) * float(wpercent)))
        desireable_size = (scale, hsize)
    else:
        # image is wider
        hpercent = scale / float(size[1])
        wsize = int((float(size[0]) * float(hpercent)))
        desireable_size = (wsize, scale)
    resized_img = cv2.resize(
        img, desireable_size[::-1], interpolation=cv2.INTER_AREA
    )  # this flips the desireable_size vector

    # Padding
    if max_ind == 0:
        # height fixed at scale, pad the width
        pad_size = scale - resized_img.shape[1]
        left = int(np.floor(pad_size / 2))
        right = int(np.ceil(pad_size / 2))
        top = int(0)
        bottom = int(0)
    else:
        # width fixed at scale, pad the height
        pad_size = scale - resized_img.shape[0]
        top = int(np.floor(pad_size / 2))
        bottom = int(np.ceil(pad_size / 2))
        left = int(0)
        right = int(0)
    resized_img = np.pad(
        resized_img, [(top, bottom), (left, right)], "constant", constant_values=0
    )

    return resized_img


def get_imgs(img_path, scale, transform=None, multiscale=False):
    x = cv2.imread(str(img_path), 0)
    # tranform images
    x = resize_img(x, scale)
    img = Image.fromarray(x).convert("RGB")
    if transform is not None:
        img = transform(img)

    return img


class MIMIC_CXR_Report_Dataset(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(os.path.join(data_path, "master.csv"))
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df["Path"] = self.df["Path"].apply(
            lambda x: os.path.join(data_path, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        self.df = self.df[self.df["split"] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=True,
                                                         mlm_probability=0.15)

        print("report sample number:", len(self.filenames))
    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "mimic_report_captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, "split")
            path = getattr(row, "Path")
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row["Path"]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

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
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        text, attention_mask = caps["input_ids"], caps["attention_mask"]
        caps_mask = self.mlm_collator(tuple(text))

        return caps_mask["input_ids"].squeeze(0), caps_mask["labels"].squeeze(0), attention_mask.squeeze(0), text.squeeze(0)

class MIMIC_CXR_Report_Dataset_name(data.Dataset):
    def __init__(self, data_path, split="train", transform=None, data_pct=1.0,
                 imsize=256, max_words=112, sent_num=3):
        super().__init__()
        if not os.path.exists(data_path):
            raise RuntimeError(f"{data_path} does not exist!")

        self.transform = transform
        self.imsize = imsize
        self.df = pd.read_csv(os.path.join(data_path, "master.csv"))
        self.df = self.df[self.df["ViewPosition"].isin(["PA", "AP"])]
        self.df["Path"] = self.df["Path"].apply(
            lambda x: os.path.join(data_path, "/".join(x.split("/")[1:])))

        # load studies and study to text mapping
        self.filenames, self.path2sent = self.load_text_data(split)

        self.df = self.df[self.df["split"] == split]
        if data_pct != 1.0 and split == "train":
            self.df = self.df.sample(frac=data_pct, random_state=42)
        self.df.reset_index(drop=True, inplace=True)

        self.tokenizer = BertTokenizer.from_pretrained(
            "emilyalsentzer/Bio_ClinicalBERT")
        self.max_words = max_words

        self.mlm_collator = DataCollatorForWholeWordMask(tokenizer=self.tokenizer, mlm=False,
                                                         mlm_probability=0.0)

        print("report sample number:", len(self.filenames))
    def load_text_data(self, split):
        # get study to captions mapping
        # TODO: check this
        filepath = os.path.join(
            BASE_DIR, "mimic_report_captions.pickle")
        if not os.path.isfile(filepath):
            print(
                f"Caption file {filepath} does not exit. Creating captions...")
            path2sent = self.create_path_2_sent_mapping()
            with open(filepath, "wb") as f:
                pickle.dump(path2sent, f, protocol=2)
                print("Save to: ", filepath)
        else:
            with open(filepath, "rb") as f:
                path2sent = pickle.load(f)

        # filter studies to use for current split
        filenames = []
        for row in self.df.itertuples():
            cur_split = getattr(row, "split")
            path = getattr(row, "Path")
            if cur_split == split and path in path2sent:
                filenames.append(path)

        return filenames, path2sent

    def create_path_2_sent_mapping(self):
        sent_lens, num_sents = [], []
        path2sent = {}
        # iterrows is not faster than itertuples ...  but it is ok
        for _, row in tqdm(self.df.iterrows(), total=self.df.shape[0]):
            # pick impression, findings, last_paragraph
            captions = ""
            captions += row["impression"]
            captions += " "
            captions += row["findings"]

            # use space instead of newline
            captions = captions.replace("\n", " ")

            # split sentences
            splitter = re.compile("[0-9]+\.")
            captions = splitter.split(captions)
            captions = [point.split(".") for point in captions]
            captions = [sent for point in captions for sent in point]

            cnt = 0
            study_sent = []
            # create tokens from captions
            for cap in captions:
                if len(cap) == 0:
                    continue

                cap = cap.replace("\ufffd\ufffd", " ")
                # picks out sequences of alphanumeric characters as tokens
                # and drops everything else
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())
                # TODO: < 3 has instances of ['no', 'pneumothorax'], ['clear', 'lung']
                if len(tokens) <= 1:
                    continue

                # filter tokens for current sentence
                included_tokens = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        included_tokens.append(t)

                if len(included_tokens) > 0:
                    study_sent.append(" ".join(included_tokens))

                cnt += len(included_tokens)

            if cnt >= 3:
                sent_lens.append(cnt)
                num_sents.append(len(study_sent))
                path2sent[row["Path"]] = study_sent

        # get report word/setence statistics
        sent_lens = np.array(sent_lens)
        num_sents = np.array(num_sents)

        print(
            f"sent lens: {sent_lens.min()},{sent_lens.mean()},{sent_lens.max()} [{np.percentile(sent_lens, 5)}, {np.percentile(sent_lens, 95)}]"
        )
        print(
            f"num sents: {num_sents.min()},{num_sents.mean()},{num_sents.max()} [{np.percentile(num_sents, 5)}, {np.percentile(num_sents, 95)}]"
        )

        return path2sent

    def __len__(self):
        return len(self.filenames)

    def get_caption(self, path):
        series_sents = self.path2sent[path]

        if len(series_sents) == 0:
            raise Exception("no sentence for path")

        # separate different sentences
        series_sents = list(filter(lambda x: x != "", series_sents))
        sent = " ".join(series_sents)

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
        key = self.filenames[index]
        caps, cap_len = self.get_caption(key)
        text, attention_mask = caps["input_ids"], caps["attention_mask"]
        caps_mask = self.mlm_collator(tuple(text))
        return caps_mask["input_ids"].squeeze(0), caps_mask["labels"].squeeze(0), attention_mask.squeeze(0), text.squeeze(0), key



def my_collate_text(batch):
    input_ids, labels, attention_mask, text, key = zip(*batch)
    input_ids = torch.stack(input_ids, 0)
    labels = torch.stack(labels, 0)
    attention_mask = torch.stack(attention_mask, 0)
    text = torch.stack(text, 0)
    return [input_ids, labels, attention_mask, text, key]


