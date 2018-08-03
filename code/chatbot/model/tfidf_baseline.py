import pandas as pd
import numpy as np
from evaluate import recall
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data():
    # Load Data
    train_df = pd.read_csv("../data/train.csv")
    test_df = pd.read_csv("../data/test.csv")
    valid_df = pd.read_csv("../data/valid.csv")
    y_test = np.zeros(len(test_df))

    return train_df, test_df, valid_df, y_test


# Random Baseline
def predict_random(context, utterances):
    return np.random.choice(len(utterances), 10, replace=False)


