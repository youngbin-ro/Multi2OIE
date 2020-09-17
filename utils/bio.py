import torch
import numpy as np
from utils import utils

pred_tag2idx = {
    'P-B': 0, 'P-I': 1, 'O': 2
}
arg_tag2idx = {
    'A0-B': 0, 'A0-I': 1,
    'A1-B': 2, 'A1-I': 3,
    'A2-B': 4, 'A2-I': 5,
    'A3-B': 6, 'A3-I': 7,
    'O': 8,
}


def get_pred_idxs(pred_tags):
    idxs = list()
    for pred_tag in pred_tags:
        idxs.append([idx.item() for idx in (pred_tag != 2).nonzero()])
    return idxs


def get_pred_mask(tensor):
    """
    Generate predicate masks by converting predicate index with 'O' tag to 1.
    Other indexes are converted to 0 which means non-masking.

    :param tensor: predicate tagged tensor with the shape of (B, L),
        where B is the batch size, L is the sequence length.
    :return: masked binary tensor with the same shape.
    """
    res = tensor.clone()
    res[tensor == pred_tag2idx['O']] = 1
    res[tensor != pred_tag2idx['O']] = 0
    return torch.tensor(res, dtype=torch.bool, device=tensor.device)


def filter_pred_tags(pred_tags, tokens):
    """
    Filter useless tokens by converting them into 'Outside' tag.
    We treat 'Inside' tag before 'Beginning' tag as meaningful signal,
    so changed them to 'Beginning' tag unlike [Stanovsky et al., 2018].

    :param pred_tags: predicate tags with the shape of (B, L).
    :param tokens: list format sentence pieces with the shape of (B, L)
    :return: tensor of filtered predicate tags with the same shape.
    """
    assert len(pred_tags) == len(tokens)
    assert len(pred_tags[0]) == len(tokens[0])

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for pred_idx, cur_tokens in enumerate(tokens):
        for tag_idx, token in enumerate(cur_tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                pred_tags[pred_idx][tag_idx] = pred_tag2idx['O']

    # filter by tags
    pred_copied = pred_tags.clone()
    for pred_idx, cur_pred_tag in enumerate(pred_copied):
        flag = False
        tag_copied = cur_pred_tag.clone()
        for tag_idx, tag in enumerate(tag_copied):
            if not flag and tag == pred_tag2idx['P-B']:
                flag = True
            elif not flag and tag == pred_tag2idx['P-I']:
                pred_tags[pred_idx][tag_idx] = pred_tag2idx['P-B']
                flag = True
            elif flag and tag == pred_tag2idx['O']:
                flag = False
    return pred_tags


def filter_arg_tags(arg_tags, pred_tags, tokens):
    """
    Same as the description of @filter_pred_tags().

    :param arg_tags: argument tags with the shape of (B, L).
    :param pred_tags: predicate tags with the same shape.
        It is used to force predicate position to be allocated the 'Outside' tag.
    :param tokens: list of string tokens with the length of L.
        It is used to force special tokens like [CLS] to be allocated the 'Outside' tag.
    :return: tensor of filtered argument tags with the same shape.
    """

    # filter by tokens ([CLS], [SEP], [PAD] tokens should be allocated as 'O')
    for arg_idx, cur_arg_tag in enumerate(arg_tags):
        for tag_idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]']:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx['O']

    # filter by tags
    arg_copied = arg_tags.clone()
    for arg_idx, (cur_arg_tag, cur_pred_tag) in enumerate(zip(arg_copied, pred_tags)):
        pred_idxs = [idx[0].item() for idx
                     in (cur_pred_tag != pred_tag2idx['O']).nonzero()]
        arg_tags[arg_idx][pred_idxs] = arg_tag2idx['O']
        cur_arg_copied = arg_tags[arg_idx].clone()
        flag_idx = 999
        for tag_idx, tag in enumerate(cur_arg_copied):
            if tag == arg_tag2idx['O']:
                flag_idx = 999
                continue
            arg_n = tag // 2  # 0: A0 / 1: A1 / ...
            inside = tag % 2  # 0: begin / 1: inside
            if not inside and flag_idx != arg_n:
                flag_idx = arg_n
            # connect_args
            elif not inside and flag_idx == arg_n:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx[f'A{arg_n}-I']
            elif inside and flag_idx != arg_n:
                arg_tags[arg_idx][tag_idx] = arg_tag2idx[f'A{arg_n}-B']
                flag_idx = arg_n
    return arg_tags


def get_max_prob_args(arg_tags, arg_probs):
    """
    Among predicted argument tags, remain only arguments with highest probs.
    The comparison of probability is made only between the same argument labels.

    :param arg_tags: argument tags with the shape of (B, L).
    :param arg_probs: argument softmax probabilities with the shape of (B, L, T),
        where B is the batch size, L is the sequence length, and T is the # of tag labels.
    :return: tensor of filtered argument tags with the same shape.
    """
    for cur_arg_tag, cur_probs in zip(arg_tags, arg_probs):
        cur_tag_probs = [cur_probs[idx][tag] for idx, tag in enumerate(cur_arg_tag)]
        for arg_n in range(4):
            b_tag = arg_tag2idx[f"A{arg_n}-B"]
            i_tag = arg_tag2idx[f"A{arg_n}-I"]
            flag = False
            total_tags = []
            cur_tags = []
            for idx, tag in enumerate(cur_arg_tag):
                if not flag and tag == b_tag:
                    flag = True
                    cur_tags.append(idx)
                elif flag and tag == i_tag:
                    cur_tags.append(idx)
                elif flag and tag == b_tag:
                    total_tags.append(cur_tags)
                    cur_tags = [idx]
                elif tag != b_tag or tag != i_tag:
                    total_tags.append(cur_tags)
                    cur_tags = []
                    flag = False
            max_idxs, max_prob = None, 0.0
            for idxs in total_tags:
                all_probs = [cur_tag_probs[idx].item() for idx in idxs]
                if len(all_probs) == 0:
                    continue
                cur_prob = all_probs[0]
                if cur_prob > max_prob:
                    max_prob = cur_prob
                    max_idxs = idxs
            if max_idxs is None:
                continue
            del_idxs = [idx for idx, tag in enumerate(cur_arg_tag)
                        if (tag in [b_tag, i_tag]) and (idx not in max_idxs)]
            cur_arg_tag[del_idxs] = arg_tag2idx['O']
    return arg_tags


def get_single_predicate_idxs(pred_tags):
    """
    Divide each single batch based on predicted predicates.
    It is necessary for predicting argument tags with specific predicate.

    :param pred_tags: tensor of predicate tags with the shape of (B, L)
        EX >>> tensor([[2, 0, 0, 1, 0, 1, 0, 2, 2, 2],
                       [2, 2, 2, 0, 1, 0, 1, 2, 2, 2],
                       [2, 2, 2, 2, 2, 2, 2, 2, 0, 1]])

    :return: list of tensors with the shape of (B, P, L)
        the number P can be different for each batch.
        EX >>> [tensor([[2., 0., 2., 2., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 0., 1., 2., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 0., 1., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 2., 0., 2., 2., 2.]]),
                tensor([[2., 2., 2., 0., 1., 2., 2., 2., 2., 2.],
                        [2., 2., 2., 2., 2., 0., 1., 2., 2., 2.]]),
                tensor([[2., 2., 2., 2., 2., 2., 2., 2., 0., 1.]])]
    """
    total_pred_tags = []
    for cur_pred_tag in pred_tags:
        cur_sent_preds = []
        begin_idxs = [idx[0].item() for idx in (cur_pred_tag == pred_tag2idx['P-B']).nonzero()]
        for i, b_idx in enumerate(begin_idxs):
            cur_pred = np.full(cur_pred_tag.shape[0], pred_tag2idx['O'])
            cur_pred[b_idx] = pred_tag2idx['P-B']
            if i == len(begin_idxs) - 1:
                end_idx = cur_pred_tag.shape[0]
            else:
                end_idx = begin_idxs[i + 1]
            for j, tag in enumerate(cur_pred_tag[b_idx:end_idx]):
                if tag.item() == pred_tag2idx['O']:
                    break
                elif tag.item() == pred_tag2idx['P-I']:
                    cur_pred[b_idx + j] = pred_tag2idx['P-I']
            cur_sent_preds.append(cur_pred)
        total_pred_tags.append(cur_sent_preds)
    return [torch.Tensor(pred_tags) for pred_tags in total_pred_tags]


def get_tuple(sentence, pred_tags, arg_tags, tokenizer):
    """
    Get string format tuples from given predicate indexes and argument tags.

    :param sentence: string format raw sentence.
    :param pred_tags: tensor of predicate tags with the shape of (# of predicates, sequence length).
    :param arg_tags: tensor of argument tags with the same shape.
    :param tokenizer: transformer BertTokenizer (bert-base-cased or bert-base-multilingual-cased)

    :return extractions: list of strings each element means predicate, arg0, arg1, ...
    :return extraction_idxs: list of indexes of each argument for calculating confidence score.
    """
    word2piece = utils.get_word2piece(sentence, tokenizer)
    words = sentence.split(' ')
    assert pred_tags.shape[0] == arg_tags.shape[0]  # number of predicates

    pred_tags = pred_tags.tolist()
    arg_tags = arg_tags.tolist()
    extractions = list()
    extraction_idxs = list()

    # loop for each predicate
    for cur_pred_tag, cur_arg_tags in zip(pred_tags, arg_tags):
        cur_extraction = list()
        cur_extraction_idxs = list()

        # get predicate
        pred_labels = [pred_tag2idx['P-B'], pred_tag2idx['P-I']]
        cur_predicate_idxs = [idx for idx, tag in enumerate(cur_pred_tag) if tag in pred_labels]
        if len(cur_predicate_idxs) == 0:
            predicates_str = ''
        else:
            cur_pred_words = list()
            for word_idx, piece_idxs in word2piece.items():
                if set(piece_idxs) <= set(cur_predicate_idxs):
                    cur_pred_words.append(word_idx)
            if len(cur_pred_words) == 0:
                predicates_str = ''
                cur_predicate_idxs = list()
            else:
                predicates_str = ' '.join([words[idx] for idx in cur_pred_words])
        cur_extraction.append(predicates_str)
        cur_extraction_idxs.append(cur_predicate_idxs)

        # get arguments
        for arg_n in range(4):
            cur_arg_labels = [arg_tag2idx[f'A{arg_n}-B'], arg_tag2idx[f'A{arg_n}-I']]
            cur_arg_idxs = [idx for idx, tag in enumerate(cur_arg_tags) if tag in cur_arg_labels]
            if len(cur_arg_idxs) == 0:
                cur_arg_str = ''
            else:
                cur_arg_words = list()
                for word_idx, piece_idxs in word2piece.items():
                    if set(piece_idxs) <= set(cur_arg_idxs):
                        cur_arg_words.append(word_idx)
                if len(cur_arg_words) == 0:
                    cur_arg_str = ''
                    cur_arg_idxs = list()
                else:
                    cur_arg_str = ' '.join([words[idx] for idx in cur_arg_words])
            cur_extraction.append(cur_arg_str)
            cur_extraction_idxs.append(cur_arg_idxs)
        extractions.append(cur_extraction)
        extraction_idxs.append(cur_extraction_idxs)
    return extractions, extraction_idxs


def get_confidence_score(pred_probs, arg_probs, extraction_idxs):
    """
    get the confidence score of each extraction for drawing PR-curve.

    :param pred_probs: (sequence length, # of predicate labels)
    :param arg_probs: (# of predicates, sequence length, # of argument labels)
    :param extraction_idxs: [[[2, 3, 4], [0, 1], [9, 10]], [[0, 1, 2], [7, 8], [4, 5]], ...]
    """
    confidence_scores = list()
    for cur_arg_prob, cur_ext_idxs in zip(arg_probs, extraction_idxs):
        if len(cur_ext_idxs[0]) == 0:
            confidence_scores.append(0)
            continue
        cur_score = 0

        # predicate score
        pred_score = max(pred_probs[cur_ext_idxs[0][0]]).item()
        cur_score += pred_score

        # argument score
        for arg_idx in cur_ext_idxs[1:]:
            if len(arg_idx) == 0:
                continue
            begin_idxs = _find_begins(arg_idx)
            arg_score = np.mean([max(cur_arg_prob[cur_idx]).item() for cur_idx in begin_idxs])
            cur_score += arg_score
        confidence_scores.append(cur_score)
    return confidence_scores


def _find_begins(idxs):
    found_begins = [idxs[0]]
    cur_flag_idx = idxs[0]
    for cur_idx in idxs[1:]:
        if cur_idx - cur_flag_idx != 1:
            found_begins.append(cur_idx)
        cur_flag_idx = cur_idx
    return found_begins
