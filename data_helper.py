import pandas as pd
import re


def clean_str(sentence):
    """
    Remove non alphabet characters and split the string into array.
    """
    return [elem for elem in [re.sub("[^a-zA-Z]", "", elem) for elem in sentence.split(' ')] if len(elem) > 0]


def get_data(path):
    """
    Read the data as dataframe.
    Make the messages string arrays.
    Return labels and messages.
    """
    data = pd.read_csv(path)

    labels = data['v1'].tolist()
    messages = data['v2'].apply(lambda x: clean_str(x)).tolist()

    return labels, messages
