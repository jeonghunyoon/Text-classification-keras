import pandas as pd
import numpy as np
import re

from collections import Counter

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.pipeline import Pipeline


def clean_str(sentence):
    """Remove non alphabet characters and split the string into array."""
    return [elem for elem in [re.sub("[^a-zA-Z]", "", elem) for elem in sentence.split(' ')] if len(elem) > 0]


def get_data(path):
    """
    Read the data as dataframe. Make the messages string arrays.
    Return labels and messages.
    """
    data = pd.read_csv(path, encoding='latin-1')

    labels = data['v1'].tolist()
    messages = data['v2'].apply(lambda x: clean_str(x)).tolist()

    return labels, messages


def pad_str(messages):
    """Pad the sentences with the maximum length of sentence."""
    max_len = max([len(message) for message in messages])
    pad_word = '<PAD>'
    result = []
    for message in messages:
        num_pad = max_len - len(message)
        result.append(np.array(message + [pad_word] * num_pad))
    return np.array(result)


def build_voca(messages):
    """Build vocabulary lookup tables."""
    corpus = Counter(messages.flatten()).most_common()
    voca_lookup = {}
    rev_voca_lookup = {}
    for idx, val in enumerate(corpus):
        voca_lookup[val[0]] = idx
        rev_voca_lookup[idx] = val[0]
    return voca_lookup, rev_voca_lookup


def get_input_seqs(messages, voca_lookup):
    """Build integer sequences for input."""
    return np.array([np.array([voca_lookup[elem] for elem in message]) for message in messages])


class ReshapeLabelEncoder(LabelEncoder):
    """For using Pipeline class, we transform the return shape."""
    def fit_transform(self, y, *args, **kwargs):
        return super().fit_transform(y).reshape(-1, 1)


def get_onehot_labels(labels):
    """Return onehot vectors for labels."""
    reshape_label_encoder = ReshapeLabelEncoder()
    onehot_encoder = OneHotEncoder()
    pipeline = Pipeline([
        ('label_encoder', reshape_label_encoder),
        ('onehot_encoder', onehot_encoder)
    ])

    onehot_vector = pipeline.fit_transform(labels)

    return onehot_vector.toarray()
