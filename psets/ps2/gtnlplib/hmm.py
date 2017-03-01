from gtnlplib.preproc import conll_seq_generator
from gtnlplib.constants import START_TAG, TRANS, END_TAG, EMIT, OFFSET
from gtnlplib import naive_bayes, most_common
import numpy as np
from collections import defaultdict

def hmm_features(tokens,curr_tag,prev_tag,m):
    """Feature function for HMM that returns emit and transition features

    :param tokens: list of tokens 
    :param curr_tag: current tag
    :param prev_tag: previous tag
    :param i: index of token to be tagged
    :returns: dict of features and counts
    :rtype: dict

    """
    ret = {}
    if m < len(tokens):
        ret[(curr_tag, tokens[m], EMIT)] = 1
    ret[(curr_tag, prev_tag, TRANS)] = 1
    return ret
    

def compute_HMM_weights(trainfile,smoothing):
    """Compute all weights for the HMM

    :param trainfile: training file
    :param smoothing: float for smoothing of both probability distributions
    :returns: defaultdict of weights, list of all possible tags (types)
    :rtype: defaultdict, list

    """
    # hint: these are your first two lines
    tag_trans_counts = most_common.get_tag_trans_counts(trainfile)
    tag_word_counts = most_common.get_tag_word_counts(trainfile)
    all_tags = tag_trans_counts.keys()

    # hint: call compute_transition_weights
    # hint: set weights for illegal transitions to -np.inf
    # hint: call get_tag_word_counts and estimate_nb_tagger
    # hint: Counter.update() combines two Counters

    # hint: return weights, all_tags
    trans_weights = compute_transition_weights(tag_trans_counts, smoothing)
    emission_weights = naive_bayes.estimate_nb_tagger(tag_word_counts, smoothing)
    tagged_emission_weights = {}
    for weight in emission_weights:
        tagged_emission_weights[(weight[0], weight[1], EMIT)] = emission_weights[weight]
    trans_weights.update(tagged_emission_weights)
    return trans_weights, all_tags


def compute_transition_weights(trans_counts, smoothing):
    """Compute the HMM transition weights, given the counts.
    Don't forget to assign smoothed probabilities to transitions which
    do not appear in the counts.
    
    This will also affect your computation of the denominator.

    :param trans_counts: counts, generated from most_common.get_tag_trans_counts
    :param smoothing: additive smoothing
    :returns: dict of features [(curr_tag,prev_tag,TRANS)] and weights

    """
    weights = defaultdict(float)
    dest_tags = set(trans_counts.keys())
    start_tags = set(trans_counts.keys())
    all_tags = set(trans_counts.keys())
    all_tags.add(END_TAG)
    dest_tags.remove(START_TAG)
    dest_tags.add(END_TAG)
    for start in start_tags:
        sum = 0
        for dest in dest_tags:
            sum += trans_counts[start][dest]
        denominator = sum + len(dest_tags) * smoothing
        for tag in dest_tags:
            weights[(tag, start, TRANS)] = np.log((trans_counts[start][tag] + smoothing) / denominator)
    for tag in all_tags:
        weights[(START_TAG, tag, TRANS)] = -np.inf
        weights[(tag, END_TAG, TRANS)] = -np.inf
    return weights


