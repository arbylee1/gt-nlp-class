from gtnlplib.constants import OFFSET
import numpy as np

# hint! use this.
argmax = lambda x: max(x.iteritems(), key=lambda y: y[1])[0]


def make_feature_vector(base_features, label):
    """take a counter of base features and a label; return a dict of features, corresponding to f(x,y)

    :param base_features: counter of base features
    :param label: label string
    :returns: dict of features, f(x,y)
    :rtype: dict

    """
    feature_vector = {(label, OFFSET): 1}
    for feature, count in base_features.iteritems():
        feature_vector[(label, feature)] = count
    return feature_vector


def predict(base_features, weights, labels):
    """prediction function

    :param base_features: a dictionary of base features and counts
    :param weights: a defaultdict of features and weights. features are tuples (label,base_feature).
    :param labels: a list of candidate labels
    :returns: top scoring label, scores of all labels
    :rtype: string, dict

    """
    scores = {}
    for label in labels:
        score = weights[(label, OFFSET)]
        for feature, count in base_features.iteritems():
            score += weights[(label, feature)] * count
        scores[label] = score
    return argmax(scores), scores