
import warnings
import plotly.graph_objects as go
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import ecommercedata as edata
import mymodel
from ecommercedata import text_preprocess
import torch.nn as nn
import torch

warnings.filterwarnings("ignore")


DEVICE = mymodel.DEVICE
BATCH_SIZE = 32
new_category = ['Household', 'Books',
                'Clothing & Accessories', 'Electronics', 'Other']
MAX_LEN = 166
import pickle 
with open('saved_vocab_to_int.pkl', 'rb') as f:
    vocabulary_to_int = pickle.load(f)
VOCAB_SIZE = len(vocabulary_to_int) + 1
EMBEDDING_DIM = 300
HIDDEN_DIM = 100
NUM_CLASSES = 5
LSTM_LAYERS = 1
IS_BIDIRECTIONAL = True
LR = 1e-4

model = mymodel.LSTM_Classifier(
    VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, LSTM_LAYERS, IS_BIDIRECTIONAL)
model = model.to(DEVICE)
model.load_state_dict(torch.load('new_model.pt', map_location="cpu"))
model.eval()

def predict_single(text, model=model, temp=5, inverse_category_list=new_category):
    text = text_preprocess(text)
    text_tok = edata.small_tokenize(text, vocabulary_to_int, MAX_LEN)
    splitted = text.split()
    actual_length = len(splitted)
    with torch.no_grad():
        x, h = text_tok.reshape(1, len(text_tok)).to(
            DEVICE),  model.init_hidden(1)
        out, _, attn_weights = model(x, h)
        attn_weights = attn_weights.cpu()
        temp = actual_length - 1 if temp > actual_length else temp
        _, important_tokens_base = torch.topk(attn_weights.squeeze(
            0)[:actual_length].unsqueeze(0), temp, dim=1)
        important_text = [splitted[token.to(int)] for token in important_tokens_base.squeeze(
            0) if token.to(int) in range(0, actual_length)]
        prediction = torch.argmax(out, dim=1).to(int)
        if prediction == 4:
            return inverse_category_list[prediction], []
        return inverse_category_list[prediction], important_text
