
get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import pandas as pd

# read data
data = sio.loadmat("boston.mat")
# convert to dataframe
df = pd.DataFrame.from_records(data["boston"], columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "BLACK", "LSTAT", "MEDV"])


from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=1/3)


def calculate_mse(predicted, output):
    """
    Returns the MSE
    """
    total = 0
    for y1, y2 in zip(predicted, output):
        total += ((y1 - y2) ** 2)
    return total / len(predicted)


def naive_regression(train_set, test_set):
    """
    Given the training and test set, 
    perform linear regression against a constant function adn returns MSE.
    """
    train_ones = np.ones((len(train_set), 1))
    test_ones = np.ones((len(test_set), 1))
    train_mse = calculate_mse(
        np.linalg.lstsq(train_ones, train_set["MEDV"])[0],
        list(train_set["MEDV"])
    )
    test_mse = calculate_mse(
        np.linalg.lstsq(test_ones, test_set["MEDV"])[0],
        list(test_set["MEDV"])
    )
    return train_mse, test_mse


# Average over 20 runs
train_mse_ave, test_mse_ave = 0, 0
for _ in range(20):
    train, test = train_test_split(df, test_size=1/3)
    train_mse, test_mse = naive_regression(train, test)
    train_mse_ave += train_mse
    test_mse_ave += test_mse
print(train_mse_ave / 20, test_mse_ave /20)


# f = c is the average of all the output values so that mse is reduced

### single attrubute regression
from collections import defaultdict

def get_prediction(weights, xs):
    """
    Calculates predictions for basis function (1, x)
    """
    predicted = []
    for x in xs:
        predicted.append(x * weights[0] + weights[1])
    return predicted


def single_attr_regression():
    """
    For all the attributes, perform linear regression with a bias term,
    average over 20 runs and returns the MSE for all attributes for train and test MSE.
    """
    train_mses = defaultdict(int)
    test_mses = defaultdict(int)
    for attr in df.columns[:-1]:
        for _ in range(20):
            train, test = train_test_split(df, test_size=1/3)
            xs = train[attr]
            ys = train["MEDV"]
            feature_matrix = []
            for attr_val in xs:
                feature_matrix.append([attr_val, 1])
            weight = np.linalg.lstsq(feature_matrix, ys)[0]
            train_mses[attr] += calculate_mse(get_prediction(weight, xs), ys)
            test_mses[attr] += calculate_mse(get_prediction(weight, test[attr]), test["MEDV"])
    for key in train_mses:
        train_mses[key] /= 20
        test_mses[key] /= 20
    return train_mses, test_mses


def get_predicted_for_mult(weight, xs):
    """
    Get predicted answers for multi param regresssion.
    """
    predicted = []
    for _, x in xs.iterrows():
        total = 0.0
        x = x.tolist()
        for w, x_ in zip(weight, x):
            total += x_ * w
        predicted.append(total)
    return predicted

def multi_param_linear_regression():
    """
    Perform regression with all the attributes as inputs with a bias term.
    Returns MSE for 
    """
    train_mse, test_mse = 0, 0
    for _ in range(20):
        train, test = train_test_split(df, test_size=1/3)
        ys = train["MEDV"]
        # drop the house prices from dataframe
        xs = train.drop(["MEDV"], axis=1)
        # add bias term
        xs["bias"] = np.ones(len(xs))
        weight = np.linalg.lstsq(xs, ys)[0]
        predicted = get_predicted_for_mult(weight, xs)
        train_mse += calculate_mse(predicted, ys)
        
        # test the model again test set
        ys = test["MEDV"]
        xs = test.drop(["MEDV"], axis=1)
        xs["bias"] = np.ones(len(xs))
        predicted = get_predicted_for_mult(weight, xs)
        test_mse += calculate_mse(predicted, ys)
    return train_mse / 20, test_mse / 20
multi_param_linear_regression()
