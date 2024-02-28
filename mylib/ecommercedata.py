import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import seaborn as sns
from collections import Counter
import torch
from torch.nn.utils.rnn import pad_sequence
from ecommercetext import text_preprocess
from torch.utils.data import Dataset


class EcommerceDataset(Dataset):
    def __init__(self, x_vectorized, y):
        self.x_vectorized = x_vectorized
        self.y = y

    def __getitem__(self, index):
        return self.x_vectorized[index], self.y[index]

    def __len__(self):
        return len(self.x_vectorized)

def prepare_data(csv_path, category_dict):
    data = pd.read_csv(csv_path)
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    data.columns = ['category', 'description']
    data.replace({'category' : category_dict}, inplace=True)
    data['description'] = data['description'].apply(text_preprocess)
    data['text_len'] = [len(text.split()) for text in data['description']]
    data = data[data['text_len'] < data['text_len'].quantile(0.95)]
    data = data[data['text_len'] > 0]
    data.dropna(inplace=True)
    data.drop_duplicates(inplace=True)
    data.reset_index(drop=True, inplace=True)
    max_len = np.max(data['text_len'])
    return data, max_len

def build_vocab(documents, existing_vocab=None):
    if existing_vocab is not None:
        vocab_dict = dict(existing_vocab)
    else:
        vocab_dict = {'<unk>' : 1}
    corpus = [word for text in documents for word in text.split()]
    count_words = Counter(corpus)
    for word, count in count_words.items():
        vocab_dict[word] = vocab_dict.get(word, 1) + count
    vocab = list(vocab_dict.items())
    vocab.sort(key=lambda x: x[1], reverse=True)
    vocab_to_int = {word:i+1 for i, (word,count) in enumerate(vocab)}
    return vocab, vocab_to_int

def huge_tokenize(documents, vocab_to_int):
    text_int = []
    for text in documents:
        r = [vocab_to_int.get(word, 1) for word in text.split()]
        r = torch.Tensor(r)
        text_int.append(r)
    text_int = pad_sequence(text_int, batch_first=True)
    return text_int

def small_tokenize(text, vocab_to_int, seq_len):
    r = [vocab_to_int.get(word, 1) for word in text.split()]
    if len(r) <= seq_len:
        zeros = list(torch.zeros(seq_len - len(r)))
        new = r + zeros
    else:
        new = r[: seq_len]
    return torch.Tensor(new).to(int)

def train_test_indices(data, train_coef=0.8):
    data_size = len(data)
    indexs = [idx for idx in range(data_size)]
    train_indices, test_indices = [], []
    np.random.shuffle(indexs)
    train_size = int(data_size * train_coef)
    for i in range(data_size):
        if i < train_size:
            train_indices.append(indexs[i])
        else:
            test_indices.append(indexs[i])
    return train_indices, test_indices

