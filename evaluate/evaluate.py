#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 19:24:30 2018

@author: longzhan
"""

import string
import numpy as np
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import re
import logging
import sys
logging.basicConfig(level=logging.INFO)

from evaluate.generalReader import GeneralReader
from evaluate.goldReader import GoldReader
from evaluate.gold_relabel import Relabel_GoldReader
from evaluate.matcher import Matcher
from operator import itemgetter


class Benchmark:
    ''' Compare the gold OIE dataset against a predicted equivalent '''
    def __init__(self, gold_fn):
        ''' Load gold Open IE, this will serve to compare against using the compare function '''
        if 'Re' in gold_fn:
            gr = Relabel_GoldReader()
        else:
            gr = GoldReader()
        gr.read(gold_fn)
        self.gold = gr.oie

    def compare(self, predicted, matchingFunc, output_fn, error_file = None):
        ''' Compare gold against predicted using a specified matching function.
            Outputs PR curve to output_fn '''

        y_true = []
        y_scores = []
        errors = []

        correctTotal = 0
        unmatchedCount = 0

        non_sent = 0
        non_match = 0

        predicted = Benchmark.normalizeDict(predicted)
        gold = Benchmark.normalizeDict(self.gold)

        for sent, goldExtractions in gold.items():
            if sent not in predicted:
                # The extractor didn't find any extractions for this sentence
                for goldEx in goldExtractions:
                    non_sent += 1
                    unmatchedCount += len(goldExtractions)
                    correctTotal += len(goldExtractions)
                continue

            predictedExtractions = predicted[sent]

            for goldEx in goldExtractions:
                correctTotal += 1
                found = False

                for predictedEx in predictedExtractions:
                    if output_fn in predictedEx.matched:
                        # This predicted extraction was already matched against a gold extraction
                        # Don't allow to match it again
                        continue

                    if matchingFunc(goldEx,
                                    predictedEx,
                                    ignoreStopwords=True,
                                    ignoreCase=True):

                        y_true.append(1)
                        y_scores.append(predictedEx.confidence)
                        predictedEx.matched.append(output_fn)
                        found = True
                        break

                if not found:
                    non_match += 1
                    errors.append(goldEx.index)
                    unmatchedCount += 1

            for predictedEx in [x for x in predictedExtractions if (output_fn not in x.matched)]:
                # Add false positives
                y_true.append(0)
                y_scores.append(predictedEx.confidence)

        y_true = y_true
        y_scores = y_scores

        print("non_sent: ", non_sent)
        print("non_match: ", non_match)
        print("correctTotal: ", correctTotal)
        print("unmatchedCount: ", unmatchedCount)

        # recall on y_true, y  (r')_scores computes |covered by extractor| / |True in what's covered by extractor|
        # to get to true recall we do:
        # r' * (|True in what's covered by extractor| / |True in gold|) = |true in what's covered| / |true in gold|
        (p, r), optimal = Benchmark.prCurve(np.array(y_true), np.array(y_scores),
                                            recallMultiplier = ((correctTotal - unmatchedCount)/float(correctTotal)))
        cur_auc = auc(r, p)
        print("AUC: {}\n Optimal (precision, recall, F1, threshold): {}".format(cur_auc, optimal))

        # Write error log to file
        if error_file:
            logging.info("Writing {} error indices to {}".format(len(errors),
                                                                 error_file))
            with open(error_file, 'w') as fout:
                fout.write('\n'.join([str(error)
                                     for error
                                      in errors]) + '\n')

        # write PR to file
        with open(output_fn, 'w') as fout:
            fout.write('{0}\t{1}\n'.format("Precision", "Recall"))
            for cur_p, cur_r in sorted(zip(p, r), key = lambda cur: cur[1]):
                fout.write('{0}\t{1}\n'.format(cur_p, cur_r))

        return optimal[:-1], cur_auc

    @staticmethod
    def prCurve(y_true, y_scores, recallMultiplier):
        # Recall multiplier - accounts for the percentage examples unreached
        # Return (precision [list], recall[list]), (Optimal F1, Optimal threshold)
        y_scores = [score \
                    if not (np.isnan(score) or (not np.isfinite(score))) \
                    else 0
                    for score in y_scores]
        
        precision_ls, recall_ls, thresholds = precision_recall_curve(y_true, y_scores)
        recall_ls = recall_ls * recallMultiplier
        optimal = max([(precision, recall, f_beta(precision, recall, beta = 1), threshold)
                       for ((precision, recall), threshold)
                       in zip(zip(precision_ls[:-1], recall_ls[:-1]),
                              thresholds)],
                       key=itemgetter(2))  # Sort by f1 score

        return ((precision_ls, recall_ls),
                optimal)

    # Helper functions:
    @staticmethod
    def normalizeDict(d):
        return dict([(Benchmark.normalizeKey(k), v) for k, v in d.items()])

    @staticmethod
    def normalizeKey(k):
        return Benchmark.removePunct(Benchmark.PTB_unescape(k.replace(' ','')))

    @staticmethod
    def PTB_escape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(u, e)
        return s

    @staticmethod
    def PTB_unescape(s):
        for u, e in Benchmark.PTB_ESCAPES:
            s = s.replace(e, u)
        return s

    @staticmethod
    def removePunct(s):
        return Benchmark.regex.sub('', s)

    # CONSTANTS
    regex = re.compile('[%s]' % re.escape(string.punctuation))

    # Penn treebank bracket escapes
    # Taken from: https://github.com/nlplab/brat/blob/master/server/src/gtbtokenize.py
    PTB_ESCAPES = [('(', '-LRB-'),
                   (')', '-RRB-'),
                   ('[', '-LSB-'),
                   (']', '-RSB-'),
                   ('{', '-LCB-'),
                   ('}', '-RCB-'),]


def f_beta(precision, recall, beta = 1):
    """
    Get F_beta score from precision and recall.
    """
    beta = float(beta) # Make sure that results are in float
    return (1 + pow(beta, 2)) * (precision * recall) / ((pow(beta, 2) * precision) + recall)


f1 = lambda precision, recall: f_beta(precision, recall, beta = 1)

#gold_flag = sys.argv[1] # to choose whether to use OIE2016 or Re-OIE2016
#in_path = sys.argv[2] # input file
#out_path = sys.argv[3] # output file

if __name__ == '__main__':
    
    gold = gold_flag
    matchingFunc = Matcher.lexicalMatch
    error_fn = "error.txt"
        
    if gold == "old":
        gold_fn = "OIE2016.txt"
    else:
        gold_fn = "Re-OIE2016.json"
    
    b = Benchmark(gold_fn) 
    s_fn = in_path
    p = GeneralReader()
    other_p = GeneralReader()
    other_p.read(s_fn)
    b.compare(predicted = other_p.oie,
              matchingFunc = matchingFunc,
              output_fn = out_path,
              error_file = error_fn)
