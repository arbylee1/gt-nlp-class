from itertools import izip

from gtnlplib.preproc import get_corpus_counts
from gtnlplib.constants import OFFSET
from gtnlplib import clf_base, evaluation

import numpy as np
from collections import defaultdict, Set


def get_corpus_counts(x,y,label):
    """Compute corpus counts of words for all documents with a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label for corpus counts
    :returns: defaultdict of corpus counts
    :rtype: defaultdict

    """
    corpus_counts = defaultdict(float)
    for i in range(len(x)):
        if y[i] == label:
            for word, count in x[i].iteritems():
                corpus_counts[word] += count
    return corpus_counts


def estimate_pxy(x,y,label,smoothing,vocab):
    """Compute smoothed log-probability P(word | label) for a given label.

    :param x: list of counts, one per instance
    :param y: list of labels, one per instance
    :param label: desired label
    :param smoothing: additive smoothing amount
    :param vocab: list of words in vocabulary
    :returns: defaultdict of log probabilities per word
    :rtype: defaultdict of log probabilities per word

    """
    counts = get_corpus_counts(x,y,label)
    sum_counts = sum(counts.values())
    divisor = sum_counts + len(vocab)*smoothing
    for word in vocab:
        counts[word] += smoothing
        counts[word] /= divisor
        counts[word] = np.log(counts[word])
    return counts

def estimate_nb(x,y,smoothing):
    """estimate a naive bayes model

    :param x: list of dictionaries of base feature counts
    :param y: list of labels
    :param smoothing: smoothing constant
    :returns: weights
    :rtype: defaultdict 

    """
    label_count = defaultdict(float)
    for label in y:
        label_count[label] += 1
    for label in label_count.iterkeys():
        label_count[label] /= len(y)
    labels = set(y)
    weights = defaultdict(float)
    vocab = set()
    for instance in x:
        vocab |= set(instance.keys())
    for label in labels:
        pxy = estimate_pxy(x, y, label, smoothing, list(vocab))
        for word in vocab:
            weights[(label, word)] = pxy[word]
        weights[(label, OFFSET)] = np.log(label_count[label])
    return weights


    
def find_best_smoother(x_tr,y_tr,x_dv,y_dv,smoothers):
    """find the smoothing value that gives the best accuracy on the dev data

    :param x_tr: training instances
    :param y_tr: training labels
    :param x_dv: dev instances
    :param y_dv: dev labels
    :param smoothers: list of smoothing values to try
    :returns: best smoothing value, scores of all smoothing values
    :rtype: float, dict

    """
    labels = set(y_tr)
    smoother_scores = {}
    for smoother in smoothers:
        nb = estimate_nb(x_dv, y_dv, smoother)
        predictions = clf_base.predict_all(x_tr, nb, list(labels))
        score = 0
        for prediction, target in izip(predictions, y_tr):
            if prediction == target:
                score+= 1
        smoother_scores[smoother] = score
    return clf_base.argmax(smoother_scores), smoother_scores