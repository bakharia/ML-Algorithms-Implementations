# coding: utf-8

import pandas as pd
import numpy as np
import random
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, accuracy_score

# 1. Train-Test-Split
def train_test_split(df, test_size):
    
    if isinstance(test_size, float):
        test_size = round(test_size * len(df))

    indices = df.index.tolist()
    test_indices = random.sample(population=indices, k=test_size)

    test_df = df.loc[test_indices]
    train_df = df.drop(test_indices)
    
    return train_df, test_df


# 2. Distinguish categorical and continuous features
def determine_type_of_feature(df):
    
    feature_types = []
    n_unique_values_treshold = 15
    for feature in df.columns:
        if feature != "label":
            unique_values = df[feature].unique()
            example_value = unique_values[0]

            if (isinstance(example_value, str)) or (len(unique_values) <= n_unique_values_treshold):
                feature_types.append("categorical")
            else:
                feature_types.append("continuous")
    
    return feature_types


# 3. Accuracy
def calculate_accuracy(predictions, labels):
    # predictions_correct = predictions == labels
    # accuracy = predictions_correct.mean()
    
    # return accuracy
    return accuracy_score(labels, predictions)


def cross_validation(df:pd.DataFrame(), n_folds):
    
    fold_size = int(len(df) // n_folds)
    df_copy = df.copy()
    data_split = []
    for i in range(n_folds):
        fold = pd.DataFrame(columns=df.columns)
        while len(fold) < fold_size:
            index = np.random.choice(df.index)
            #fold.append(df_copy.loc[df.index == index])
            fold = pd.concat([fold, df_copy.loc[df_copy.index == index]])
            df_copy = df_copy.loc[df_copy.index != index]
        data_split.append(fold)
    print([len(x) for x in data_split])
    return data_split

def calculate_precision(predictions, labels):
    return precision_score(labels, predictions, average='macro')

def calculate_recall(predictions, labels):
    return recall_score(labels, predictions, average='macro')

def calculate_f1score(predictions, labels):
    return f1_score(labels, predictions, average='macro')

def calculate_confusionMatrix(predictions, labels):
    return confusion_matrix(labels, predictions)