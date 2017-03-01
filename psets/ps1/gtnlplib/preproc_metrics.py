def get_token_type_ratio(vocabulary):
    """compute the ratio of tokens to types

    :param vocabulary: a Counter of words and their frequencies
    :returns: ratio of tokens to types
    :rtype: float

    """
    types = 0
    tokens = 0
    for key, val in vocabulary.iteritems():
        types += 1
        tokens += val
    return float(tokens)/types

def type_frequency(vocabulary, k):
    """compute the number of words that occur exactly k times

    :param vocabulary: a Counter of words and their frequencies
    :param k: desired frequency
    :returns: number of words appearing k times
    :rtype: int

    """
    tokens = 0
    for key, val in vocabulary.iteritems():
        if val == k:
            tokens += 1
    return tokens

def unseen_types(first_vocab, second_vocab):
    """compute the number of words that appear in the second vocab but not in the first vocab

    :param first_vocab: a Counter of words and their frequencies in one dataset
    :param second_vocab: a Counter of words and their frequencies in another dataset
    :returns: number of words that appear in the second dataset but not  in the first dataset
    :rtype: int

    """
    unseen = 0
    for key in second_vocab.keys():
        if first_vocab[key] == 0:
            unseen += 1
    return unseen
