import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from utils import utils
from transformers import BertTokenizer
from utils.bio import pred_tag2idx, arg_tag2idx


def load_data(data_path,
              batch_size,
              max_len=64,
              train=True,
              tokenizer_config='bert-base-cased'):
    if train:
        return DataLoader(
            dataset=OieDataset(
                data_path,
                max_len,
                tokenizer_config),
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
            drop_last=True)
    else:
        return DataLoader(
            dataset=OieEvalDataset(
                data_path,
                max_len,
                tokenizer_config),
            batch_size=batch_size,
            num_workers=4,
            pin_memory=True)


class OieDataset(Dataset):
    def __init__(self, data_path, max_len=64, tokenizer_config='bert-base-cased'):
        data = utils.load_pkl(data_path)
        self.tokens = data['tokens']
        self.single_pred_labels = data['single_pred_labels']
        self.single_arg_labels = data['single_arg_labels']
        self.all_pred_labels = data['all_pred_labels']

        self.max_len = max_len
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
        self.vocab = self.tokenizer.vocab

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']

    def add_pad(self, token_ids):
        diff = self.max_len - len(token_ids)
        if diff > 0:
            token_ids += [self.pad_idx] * diff
        else:
            token_ids = token_ids[:self.max_len-1] + [self.sep_idx]
        return token_ids

    def add_special_token(self, token_ids):
        return [self.cls_idx] + token_ids + [self.sep_idx]

    def idx2mask(self, token_ids):
        return [token_id != self.pad_idx for token_id in token_ids]

    def add_pad_to_labels(self, pred_label, arg_label, all_pred_label):
        pred_outside = np.array([pred_tag2idx['O']])
        arg_outside = np.array([arg_tag2idx['O']])

        pred_label = np.concatenate([pred_outside, pred_label, pred_outside])
        arg_label = np.concatenate([arg_outside, arg_label, arg_outside])
        all_pred_label = np.concatenate([pred_outside, all_pred_label, pred_outside])

        diff = self.max_len - pred_label.shape[0]
        if diff > 0:
            pred_pad = np.array([pred_tag2idx['O']] * diff)
            arg_pad = np.array([arg_tag2idx['O']] * diff)
            pred_label = np.concatenate([pred_label, pred_pad])
            arg_label = np.concatenate([arg_label, arg_pad])
            all_pred_label = np.concatenate([all_pred_label, pred_pad])
        elif diff == 0:
            pass
        else:
            pred_label = np.concatenate([pred_label[:-1], pred_outside])
            arg_label = np.concatenate([arg_label[:-1], arg_outside])
            all_pred_label = np.concatenate([all_pred_label[:-1], pred_outside])
        return [pred_label, arg_label, all_pred_label]

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        token_ids = self.tokenizer.convert_tokens_to_ids(self.tokens[idx])
        token_ids_padded = self.add_pad(self.add_special_token(token_ids))
        att_mask = self.idx2mask(token_ids_padded)
        labels = self.add_pad_to_labels(
            self.single_pred_labels[idx],
            self.single_arg_labels[idx],
            self.all_pred_labels[idx])
        single_pred_label, single_arg_label, all_pred_label = labels

        assert len(token_ids_padded) == self.max_len
        assert len(att_mask) == self.max_len
        assert single_pred_label.shape[0] == self.max_len
        assert single_arg_label.shape[0] == self.max_len
        assert all_pred_label.shape[0] == self.max_len

        batch = [
            torch.tensor(token_ids_padded),
            torch.tensor(att_mask),
            torch.tensor(single_pred_label),
            torch.tensor(single_arg_label),
            torch.tensor(all_pred_label)
        ]
        return batch


class OieEvalDataset(Dataset):
    def __init__(self, data_path, max_len, tokenizer_config='bert-base-cased'):
        self.sentences = utils.load_pkl(data_path)
        self.tokenizer = BertTokenizer.from_pretrained(tokenizer_config)
        self.vocab = self.tokenizer.vocab
        self.max_len = max_len

        self.pad_idx = self.vocab['[PAD]']
        self.cls_idx = self.vocab['[CLS]']
        self.sep_idx = self.vocab['[SEP]']
        self.mask_idx = self.vocab['[MASK]']
        
    def add_pad(self, token_ids):
        diff = self.max_len - len(token_ids)
        if diff > 0:
            token_ids += [self.pad_idx] * diff
        else:
            token_ids = token_ids[:self.max_len-1] + [self.sep_idx]
        return token_ids

    def idx2mask(self, token_ids):
        return [token_id != self.pad_idx for token_id in token_ids]

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        token_ids = self.add_pad(self.tokenizer.encode(self.sentences[idx]))
        att_mask = self.idx2mask(token_ids)
        token_strs = self.tokenizer.convert_ids_to_tokens(token_ids)
        sentence = self.sentences[idx]

        assert len(token_ids) == self.max_len
        assert len(att_mask) == self.max_len
        assert len(token_strs) == self.max_len
        batch = [
            torch.tensor(token_ids),
            torch.tensor(att_mask),
            token_strs,
            sentence
        ]
        return batch

