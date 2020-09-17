import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from torch.nn.modules.container import ModuleList
from transformers import BertModel


class ArgModule(nn.Module):
    def __init__(self, arg_layer, n_layers):
        """
        Module for extracting arguments based on given encoder output and predicates.
        It uses ArgExtractorLayer as a base block and repeat the block N('n_layers') times

        :param arg_layer: an instance of the ArgExtractorLayer() class (required)
        :param n_layers: the number of sub-layers in the ArgModule (required).
        """
        super(ArgModule, self).__init__()
        self.layers = _get_clones(arg_layer, n_layers)
        self.n_layers = n_layers

    def forward(self, encoded, predicate, pred_mask=None):
        """
        :param encoded: output from sentence encoder with the shape of (L, B, D),
            where L is the sequence length, B is the batch size, D is the embedding dimension
        :param predicate: output from predicate module with the shape of (L, B, D)
        :param pred_mask: mask that prevents attention to tokens which are not predicates
            with the shape of (B, L)
        :return: tensor like Transformer Decoder Layer Output
        """
        output = encoded
        for layer_idx in range(self.n_layers):
            output = self.layers[layer_idx](
                target=output, source=predicate, key_mask=pred_mask)
        return output


class ArgExtractorLayer(nn.Module):
    def __init__(self,
                 d_model=768,
                 n_heads=8,
                 d_feedforward=2048,
                 dropout=0.1,
                 activation='relu'):
        """
        A layer similar to Transformer decoder without decoder self-attention.
        (only encoder-decoder multi-head attention followed by feed-forward layers)

        :param d_model: model dimensionality (default=768 from BERT-base)
        :param n_heads: number of heads in multi-head attention layer
        :param d_feedforward: dimensionality of point-wise feed-forward layer
        :param dropout: drop rate of all layers
        :param activation: activation function after first feed-forward layer
        """
        super(ArgExtractorLayer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.linear1 = nn.Linear(d_model, d_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

    def forward(self, target, source, key_mask=None):
        """
        Single Transformer Decoder layer without self-attention

        :param target: a tensor which takes a role as a query
        :param source: a tensor which takes a role as a key & value
        :param key_mask: key mask tensor with the shape of (batch_size, sequence_length)
        """
        # Multi-head attention layer (+ add & norm)
        attended = self.multihead_attn(
            target, source, source,
            key_padding_mask=key_mask)[0]
        skipped = target + self.dropout1(attended)
        normed = self.norm1(skipped)

        # Point-wise feed-forward layer (+ add & norm)
        projected = self.linear2(self.dropout2(self.activation(self.linear1(normed))))
        skipped = normed + self.dropout1(projected)
        normed = self.norm2(skipped)
        return normed


class Multi2OIE(nn.Module):
    def __init__(self,
                 bert_config='bert-base-cased',
                 mh_dropout=0.1,
                 pred_clf_dropout=0.,
                 arg_clf_dropout=0.3,
                 n_arg_heads=8,
                 n_arg_layers=4,
                 pos_emb_dim=64,
                 pred_n_labels=3,
                 arg_n_labels=9):
        super(Multi2OIE, self).__init__()
        self.pred_n_labels = pred_n_labels
        self.arg_n_labels = arg_n_labels

        self.bert = BertModel.from_pretrained(
            bert_config,
            output_hidden_states=True)
        d_model = self.bert.config.hidden_size
        self.pred_dropout = nn.Dropout(pred_clf_dropout)
        self.pred_classifier = nn.Linear(d_model, self.pred_n_labels)

        self.position_emb = nn.Embedding(3, pos_emb_dim, padding_idx=0)
        d_model += (d_model + pos_emb_dim)
        arg_layer = ArgExtractorLayer(
            d_model=d_model,
            n_heads=n_arg_heads,
            dropout=mh_dropout)
        self.arg_module = ArgModule(arg_layer, n_arg_layers)
        self.arg_dropout = nn.Dropout(arg_clf_dropout)
        self.arg_classifier = nn.Linear(d_model, arg_n_labels)

    def forward(self,
                input_ids,
                attention_mask,
                predicate_mask=None,
                predicate_hidden=None,
                total_pred_labels=None,
                arg_labels=None):

        # predicate extraction
        bert_hidden = self.bert(input_ids, attention_mask)[0]
        pred_logit = self.pred_classifier(self.pred_dropout(bert_hidden))

        # predicate loss
        if total_pred_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = pred_logit.view(-1, self.pred_n_labels)
            active_labels = torch.where(
                active_loss, total_pred_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(total_pred_labels))
            pred_loss = loss_fct(active_logits, active_labels)

        # inputs for argument extraction
        pred_feature = _get_pred_feature(bert_hidden, predicate_mask)
        position_vectors = self.position_emb(_get_position_idxs(predicate_mask, input_ids))
        bert_hidden = torch.cat([bert_hidden, pred_feature, position_vectors], dim=2)
        bert_hidden = bert_hidden.transpose(0, 1)

        # argument extraction
        arg_hidden = self.arg_module(bert_hidden, bert_hidden, predicate_mask)
        arg_hidden = arg_hidden.transpose(0, 1)
        arg_logit = self.arg_classifier(self.arg_dropout(arg_hidden))

        # argument loss
        if arg_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = arg_logit.view(-1, self.arg_n_labels)
            active_labels = torch.where(
                active_loss, arg_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(arg_labels))
            arg_loss = loss_fct(active_logits, active_labels)

        # total loss
        batch_loss = pred_loss + arg_loss
        outputs = (batch_loss, pred_loss, arg_loss)
        return outputs

    def extract_predicate(self,
                          input_ids,
                          attention_mask):
        bert_hidden = self.bert(input_ids, attention_mask)[0]
        pred_logit = self.pred_classifier(bert_hidden)
        return pred_logit, bert_hidden

    def extract_argument(self,
                         input_ids,
                         predicate_hidden,
                         predicate_mask):
        pred_feature = _get_pred_feature(predicate_hidden, predicate_mask)
        position_vectors = self.position_emb(_get_position_idxs(predicate_mask, input_ids))
        arg_input = torch.cat([predicate_hidden, pred_feature, position_vectors], dim=2)
        arg_input = arg_input.transpose(0, 1)
        arg_hidden = self.arg_module(arg_input, arg_input, predicate_mask)
        arg_hidden = arg_hidden.transpose(0, 1)
        return self.arg_classifier(arg_hidden)


class BERTBiLSTM(nn.Module):
    def __init__(self,
                 bert_config='bert-base-cased',
                 lstm_dropout=0.3,
                 pred_clf_dropout=0.,
                 arg_clf_dropout=0.3,
                 pos_emb_dim=256,
                 pred_n_labels=3,
                 arg_n_labels=9):
        super(BERTBiLSTM, self).__init__()
        self.pred_n_labels = pred_n_labels
        self.arg_n_labels = arg_n_labels

        self.bert = BertModel.from_pretrained(
            bert_config,
            output_hidden_states=True)
        d_model = self.bert.config.hidden_size
        self.pred_dropout = nn.Dropout(pred_clf_dropout)
        self.pred_classifier = nn.Linear(d_model, self.pred_n_labels)

        self.position_emb = nn.Embedding(3, pos_emb_dim, padding_idx=0)
        d_model += pos_emb_dim
        self.arg_module = nn.LSTM(
            input_size=d_model,
            hidden_size=d_model,
            num_layers=3,
            dropout=lstm_dropout,
            batch_first=True,
            bidirectional=True)
        self.arg_dropout = nn.Dropout(arg_clf_dropout)
        self.arg_classifier = nn.Linear(d_model * 2, arg_n_labels)

    def forward(self,
                input_ids,
                attention_mask,
                predicate_mask=None,
                predicate_hidden=None,
                total_pred_labels=None,
                arg_labels=None):

        # predicate extraction
        bert_hidden = self.bert(input_ids, attention_mask)[0]
        pred_logit = self.pred_classifier(self.pred_dropout(bert_hidden))

        # predicate loss
        if total_pred_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = pred_logit.view(-1, self.pred_n_labels)
            active_labels = torch.where(
                active_loss, total_pred_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(total_pred_labels))
            pred_loss = loss_fct(active_logits, active_labels)

        # argument extraction
        position_vectors = self.position_emb(_get_position_idxs(predicate_mask, input_ids))
        bert_hidden = torch.cat([bert_hidden, position_vectors], dim=2)
        arg_hidden = self.arg_module(bert_hidden)[0]
        arg_logit = self.arg_classifier(self.arg_dropout(arg_hidden))

        # argument loss
        if arg_labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1
            active_logits = arg_logit.view(-1, self.arg_n_labels)
            active_labels = torch.where(
                active_loss, arg_labels.view(-1),
                torch.tensor(loss_fct.ignore_index).type_as(arg_labels))
            arg_loss = loss_fct(active_logits, active_labels)

        # total loss
        batch_loss = pred_loss + arg_loss
        outputs = (batch_loss, pred_loss, arg_loss)
        return outputs

    def extract_predicate(self,
                          input_ids,
                          attention_mask):
        bert_hidden = self.bert(input_ids, attention_mask)[0]
        pred_logit = self.pred_classifier(bert_hidden)
        return pred_logit, bert_hidden

    def extract_argument(self,
                         input_ids,
                         predicate_hidden,
                         predicate_mask):
        position_vectors = self.position_emb(_get_position_idxs(predicate_mask, input_ids))
        arg_input = torch.cat([predicate_hidden, position_vectors], dim=2)
        arg_hidden = self.arg_module(arg_input)[0]
        return self.arg_classifier(arg_hidden)


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)


def _get_clones(module, n):
    return ModuleList([copy.deepcopy(module) for _ in range(n)])


def _get_position_idxs(pred_mask, input_ids):
    position_idxs = torch.zeros(pred_mask.shape, dtype=int, device=pred_mask.device)
    for mask_idx, cur_mask in enumerate(pred_mask):
        position_idxs[mask_idx, :] += 2
        cur_nonzero = (cur_mask == 0).nonzero()
        start = torch.min(cur_nonzero).item()
        end = torch.max(cur_nonzero).item()
        position_idxs[mask_idx, start:end + 1] = 1
        pad_start = max(input_ids[mask_idx].nonzero()).item() + 1
        position_idxs[mask_idx, pad_start:] = 0
    return position_idxs


def _get_pred_feature(pred_hidden, pred_mask):
    B, L, D = pred_hidden.shape
    pred_features = torch.zeros((B, L, D), device=pred_mask.device)
    for mask_idx, cur_mask in enumerate(pred_mask):
        pred_position = (cur_mask == 0).nonzero().flatten()
        pred_feature = torch.mean(pred_hidden[mask_idx, pred_position], dim=0)
        pred_feature = torch.cat(L * [pred_feature.unsqueeze(0)])
        pred_features[mask_idx, :, :] = pred_feature
    return pred_features

