from gtnlplib import tagger_base, constants
from collections import defaultdict

def sp_update(tokens,tags,weights,feat_func,tagger,all_tags):
    """compute the structure perceptron update for a single instance

    :param tokens: tokens to tag 
    :param tags: gold tags
    :param weights: weights
    :param feat_func: local feature function from (tokens,y_m,y_{m-1},m) --> dict of features and counts
    :param tagger: function from (tokens,feat_func,weights,all_tags) --> tag sequence
    :param all_tags: list of all candidate tags
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    feature_vect = defaultdict(float)
    predicted_tags = tagger(tokens, feat_func, weights, all_tags)[0]
    if predicted_tags != tags:
        feature_vect.update(tagger_base.compute_features(tokens, tags, feat_func))
        pred_feature_vect = tagger_base.compute_features(tokens, predicted_tags, feat_func)
        for pred_feature in pred_feature_vect:
            feature_vect[pred_feature] -= pred_feature_vect[pred_feature]
    return feature_vect
    
def estimate_perceptron(labeled_instances,feat_func,tagger,N_its,all_tags=None):
    """Estimate a structured perceptron

    :param labeled instances: list of (token-list, tag-list) tuples, each representing a tagged sentence
    :param feat_func: function from list of words and index to dict of features
    :param tagger: function from list of words, features, weights, and candidate tags to list of tags
    :param N_its: number of training iterations
    :param all_tags: optional list of candidate tags. If not provided, it is computed from the dataset.
    :returns: weight dictionary
    :returns: list of weight dictionaries at each iteration
    :rtype: defaultdict, list

    """
    """
    You can almost copy-paste your perceptron.estimate_avg_perceptron function here. 
    The key differences are:
    (1) the input is now a list of (token-list, tag-list) tuples
    (2) call sp_update to compute the update after each instance.
    """

    # compute all_tags if it's not provided
    if all_tags is None:
        all_tags = set()
        for tokens,tags in labeled_instances:
            all_tags.update(tags)

    # this initialization should make sure there isn't a tie for the first prediction
    # this makes it easier to test your code
    weights = defaultdict(float,
                          {('NOUN',constants.OFFSET):1e-3})
    change = defaultdict(lambda:1)
    avg_weights = weights.copy()
    w_sum = defaultdict(float)
    weight_history = []
    t = 0
    # the rest is up to you!
    for it in xrange(N_its):
        for instance in labeled_instances:
            t += 1
            weights_update = sp_update(instance[0], instance[1], weights, feat_func, tagger, all_tags)
            print weights_update
            for weight in weights_update:
                w_sum[weight] += weights[weight] * (t - change[weight])
                weights[weight] += weights_update[weight]
                change[weight] = t
        for weight in weights:
            w_sum[weight] += weights[weight]*(t + 1 - change[weight])
            change[weight] = t + 1
            avg_weights[weight] = w_sum[weight] / t
        weight_history.append(avg_weights.copy())
    return avg_weights, weight_history




