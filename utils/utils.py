import argparse
import torch
import os
import random
import numpy as np
import pickle
import pandas as pd
import json
import copy
from model import Multi2OIE, BERTBiLSTM
from transformers import get_linear_schedule_with_warmup, AdamW


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def clean_config(config):
    device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
    config.device = device
    config.pred_n_labels = 3
    config.arg_n_labels = 9
    os.makedirs(config.save_path, exist_ok=True)
    return config


def get_models(bert_config,
               pred_n_labels=3,
               arg_n_labels=9,
               n_arg_heads=8,
               n_arg_layers=4,
               lstm_dropout=0.3,
               mh_dropout=0.1,
               pred_clf_dropout=0.,
               arg_clf_dropout=0.3,
               pos_emb_dim=64,
               use_lstm=False,
               device=None):
    if not use_lstm:
        return Multi2OIE(
            bert_config=bert_config,
            mh_dropout=mh_dropout,
            pred_clf_dropout=pred_clf_dropout,
            arg_clf_dropout=arg_clf_dropout,
            n_arg_heads=n_arg_heads,
            n_arg_layers=n_arg_layers,
            pos_emb_dim=pos_emb_dim,
            pred_n_labels=pred_n_labels,
            arg_n_labels=arg_n_labels).to(device)
    else:
        return BERTBiLSTM(
            bert_config=bert_config,
            lstm_dropout=lstm_dropout,
            pred_clf_dropout=pred_clf_dropout,
            arg_clf_dropout=arg_clf_dropout,
            pos_emb_dim=pos_emb_dim,
            pred_n_labels=pred_n_labels,
            arg_n_labels=arg_n_labels).to(device)


def save_pkl(path, file):
    with open(path, 'wb') as f:
        pickle.dump(file, f)


def load_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def get_word2piece(sentence, tokenizer):
    words = sentence.split(' ')
    word2piece = {idx: list() for idx in range(len(words))}
    sentence_pieces = list()
    piece_idx = 1
    for word_idx, word in enumerate(words):
        pieces = tokenizer.tokenize(word)
        sentence_pieces += pieces
        for piece_idx_added, piece in enumerate(pieces):
            word2piece[word_idx].append(piece_idx + piece_idx_added)
        piece_idx += len(pieces)
    assert len(sentence_pieces) == piece_idx - 1
    return word2piece


def get_train_modules(model,
                      lr,
                      total_steps,
                      warmup_steps):
    optimizer = AdamW(
        model.parameters(), lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, warmup_steps, total_steps)
    return optimizer, scheduler


class SummaryManager:
    def __init__(self, config):
        self.config = config
        self.save_config()
        columns = ['epoch', 'train_predicate_loss', 'train_argument_loss']
        for cur_dev_path in config.dev_data_path:
            cur_dev_name = cur_dev_path.split('/')[-1].replace('.pkl', '')
            for metric in ['f1', 'prec', 'rec', 'auc', 'sum']:
                columns.append(f'{cur_dev_name}_{metric}')
        columns.append('total_sum')
        self.result_df = pd.DataFrame(columns=columns)
        self.save_df()

    def save_config(self, display=True):
        if display:
            for key, value in self.config.__dict__.items():
                print("{}: {}".format(key, value))
            print()
        copied = copy.deepcopy(self.config)
        copied.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        with open(os.path.join(copied.save_path, 'config.json'), 'w') as fp:
            json.dump(copied.__dict__, fp, indent=4)

    def save_results(self, results):
        self.result_df = pd.read_csv(os.path.join(self.config.save_path, 'train_results.csv'))
        self.result_df.loc[len(self.result_df.index)] = results
        self.save_df()

    def save_df(self):
        self.result_df.to_csv(os.path.join(self.config.save_path, 'train_results.csv'), index=False)


def set_model_name(dev_results, epoch, step=None):
    if step is not None:
        return "model-epoch{}-step{}-score{:.4f}.bin".format(epoch, step, dev_results)
    else:
        return "model-epoch{}-end-score{:.4f}.bin".format(epoch, dev_results)


def print_results(message, results, names):
    print(f"\n===== {message} =====")
    for result, name in zip(results, names):
        print("{}: {:.5f}".format(name, result))
    print()
