from collections import defaultdict
from gtnlplib.clf_base import predict,make_feature_vector,argmax

def perceptron_update(x,y,weights,labels):
    """compute the perceptron update for a single instance

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param weights: a weight vector, represented as a dict
    :param labels: set of possible labels
    :returns: updates to weights, which should be added to weights
    :rtype: defaultdict

    """
    updates = defaultdict(float)
    prediction = predict(x, weights, labels)
    if prediction[0] != y:
        feature_vector = make_feature_vector(x, y)
        y_hat_feature_vector = make_feature_vector(x, prediction[0])
        for feature in feature_vector:
            updates[feature] = feature_vector[feature]
        for feature in y_hat_feature_vector:
            updates[feature] = -y_hat_feature_vector[feature]
    return updates


def estimate_perceptron(x,y,N_its):
    """estimate perceptron weights for N_its iterations over the dataset (x,y)

    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list

    """
    labels = set(y)
    weights = defaultdict(float)
    weight_history = []
    for it in xrange(N_its):
        for x_i,y_i in zip(x,y):
            weights_update = perceptron_update(x_i, y_i, weights, labels)
            for weight in weights_update:
                weights[weight] += weights_update[weight]
        weight_history.append(weights.copy())
    return weights, weight_history


def estimate_avg_perceptron(x, y, N_its):
    """estimate averaged perceptron classifier
    :param x: instance, a counter of base features and weights
    :param y: label, a string
    :param N_its: number of iterations over the entire dataset
    :returns: weight dictionary
    :returns: list of weights dictionaries at each iteration
    :rtype: defaultdict, list
    """
    labels = set(y)
    w_sum = defaultdict(float)  # hint
    weights = defaultdict(float)
    weight_history = []

    avg_weights = defaultdict(float)
    t = 1.0  # hint
    for it in xrange(N_its):
        for x_i, y_i in zip(x, y):
            t += 1
            weights_update = perceptron_update(x_i, y_i, weights, labels)
            for weight in weights_update:
                weights[weight] += weights_update[weight]
            for weight in weights:
                w_sum[weight] += weights[weight]
        for weight in weights:
            avg_weights[weight] = w_sum[weight]/t
        weight_history.append(avg_weights.copy())
    return avg_weights, weight_history
