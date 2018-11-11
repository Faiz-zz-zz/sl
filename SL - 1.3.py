import numpy as np
from numpy import linalg as LA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from sklearn.model_selection import KFold
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import json
import pandas as pd
from sklearn.model_selection import train_test_split


class KernelRidgeRegression(object):
    """docstring fo KernelRidgeRegression."""

    def __init__(self, sigma, gama, x, y):
        self.sigma = sigma
        self.gama = gama
        self.x = x
        self.y = y
        self.K = self.applyGussianKernel()
        self.alphas = self.calculateAlphas()
        self.se = self.calculateSE()


    def calculateGussianKernel(self, xi, xj):
        dist = LA.norm(xi-xj)
        k_xi_xj = np.exp(np.divide(-(dist**2), 2*(self.sigma**2)))
        return k_xi_xj


    def calculateSE(self):
        se = np.subtract(self.y, self.K @ self.alphas)
        se_trans = np.transpose(se)
        leftmatrix = se_trans @ se
        rightmatrix = self.gama * (np.transpose(self.alphas) @ self.K @ self.alphas)
        return leftmatrix + rightmatrix


    def calculateAlphas(self):
        leftmat = np.linalg.inv(np.add(self.K, self.gama*np.identity(len(self.x))))
        return leftmat @ self.y
    
    def calculateMSE(self):
        data_size = len(self.x)
        MSE = self.gama * np.transpose(self.alphas) @ self.K @ self.alphas
        for i in range(data_size):
            alphaj_kij = 0 
            for j in range(data_size):
                alphaj_kij += self.alphas[j] * self.K[i][j]
            MSE += (alphaj_kij - self.y[i])**2

        return MSE/data_size
        

    def applyGussianKernel(self):
        k_value = [[0]*len(self.x) for _ in range(len(self.x))]
        for i in range(len(self.x)):
            for j in range(i, len(self.x)):
                k_value[i][j] = self.calculateGussianKernel(self.x[i], self.x[j])
        return np.array(k_value)



data = sio.loadmat("boston.mat")
atts = ["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "BLACK", "LSTAT"]
mse_training_over_20_samples = 0
mse_test_over_20_samples = 0
# Kernelised Ridge Regression
for i in range(20):
    print("round {} for kernel regression".format(i))
    MSE_training, MSE_test = findBestSigmaGamaAndCalcMSE(data)
    mse_training_over_20_samples += MSE_training
    mse_test_over_20_samples += MSE_test
print("Kernelised Rigde Regression")
print("training error: {}, test error: {}".format(mse_training_over_20_samples/20, mse_test_over_20_samples/20))
              

mse_training_over_20_samples = 0
mse_test_over_20_samples = 0
# Naive Regression
for i in range(20):
    print("round {} for naive regression".format(i))
    MSE_training, MSE_test = findBestSigmaGamaAndCalcMSE(data, regression_type="naive")
    mse_training_over_20_samples += MSE_training
    mse_test_over_20_samples += MSE_test
print("Naive Regression")
print("training error: {}, test error: {}".format(mse_training_over_20_samples/20, mse_test_over_20_samples/20))


mse_training_over_20_samples = 0
mse_test_over_20_samples = 0
# Linear Regression with all attributes + bias term
for i in range(20):
    print("round {} for all attr regression".format(i))
    MSE_training, MSE_test = findBestSigmaGamaAndCalcMSE(data, regression_type="all_attr")
    mse_training_over_20_samples += MSE_training
    mse_test_over_20_samples += MSE_test
print("Linear Regression with all attributes + bias term")
print("training error: {}, test error: {}".format(mse_training_over_20_samples/20, mse_test_over_20_samples/20))


for att in atts:
    mse_training_over_20_samples = 0
    mse_test_over_20_samples = 0
    for i in range(20):
        print("round {} for one attr regression for attribute {}".format(i, att))
        MSE_training, MSE_test = findBestSigmaGamaAndCalcMSE(data, regression_type="one_attr", attribute=att)
        mse_training_over_20_samples += MSE_training
        mse_test_over_20_samples += MSE_test
    print("Linear Regression with one attribute + bias term for attribute: {}".format(att))
    print("training error: {}, test error: {}".format(mse_training_over_20_samples/20, mse_test_over_20_samples/20)) 

        

def findBestSigmaGamaAndCalcMSE(data, regression_type="kernelised_ridge", attribute=None):
    x, y, x_test, y_test = splitData(data, regression_type, attribute)
    training_folds = applyKFold(x, y)
    gamas = np.array([2 ** i for i in list(range(-40, -25))])
    sigmas =  np.array([2 ** (i/2) for i in list(range(14, 27))])
    minSE = 999999999999
    best_pair = []
    cross_validation_pairs = []
    for i in range(len(gamas)):
        for j in range(len(sigmas)):
            averageSE = 0
            for X, Y in training_folds:
                krr = KernelRidgeRegression(sigmas[j], gamas[i], X, Y)
                averageSE += krr.se
            averageSE = averageSE/len(training_folds)
            cross_validation_pairs.append([averageSE, gamas[i], sigmas[j]])
            if averageSE < minSE:
                minSE = krr.se
                best_pair = [sigmas[j], gamas[i]]
    MSE_training = calcMSEforBestSigmaGama(best_pair, x, y)
    MSE_test = calcMSEforBestSigmaGama(best_pair, x_test, y_test)
    return MSE_training, MSE_test

def calcMSEforBestSigmaGama(best_pair, data_X, data_Y):
    sigma, gama = best_pair
    krr = KernelRidgeRegression(sigma, gama, data_X, data_Y)
    return krr.calculateMSE()


# the input is a list of list of the format (averageSE over folds of validation, gama, sigma)
def plot_cross_validation_error_sigma_gama(tuples):
    cross_validation_errors, gamas, sigmas = zip(*tuples)
    gamas, sigmas = list(gamas), list(sigmas)
    cves = [a.tolist()[0][0] for a in cross_validation_errors]
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(g, s, c)
    ax.set_xlabel('gama')
    ax.set_ylabel('sigma')
    ax.view_init(35, 75)



def calculateMSE(self, X, Y):
    data_size = len(data)
    MSE = self.gama @ np.transpose(self.alphas) @ self.K @ self.alphas
    for i in range(data_size):
        alphaj_kij = 0 
        for j in range(data_size):
            alphaj_kij += self.alphas[j] * self.K[i][j]
        MSE += (alphaj_kij - self.y[i])**2

    return MSE/data_size"


# This method loads the boston house data and splits the data to 2/3 for training and 1/3 for testing.
def splitData(data, regression_type="kernelised_ridge", attribute=None):
    df = pd.DataFrame.from_records(data["boston"], columns=["CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "BLACK", "LSTAT", "MEDV"])
    df['bias'] = np.ones((len(df), 1))
    df_train_X, df_test_X = train_test_split(df, test_size=1/3)
    df_train_Y, df_test_Y = df_train_X['MEDV'], df_test_X['MEDV']
    del df_train_X['MEDV']
    del df_test_X['MEDV']
    if regression_type == "kernelised_ridge":
        del df_train_X['bias']
        del df_test_X['bias']
        df_train_X, df_train_Y = df_train_X.as_matrix(), df_train_Y.as_matrix()
        df_test_X, df_test_Y = df_test_X.as_matrix(), df_test_Y.as_matrix()
    elif regression_type == "naive":
        df_train_X = np.ones((len(df_train_X), 1))
        df_test_X = np.ones((len(df_test_X), 1))
        df_train_Y, df_test_Y = df_train_Y.as_matrix(), df_test_Y.as_matrix()
    elif regression_type == "all_attr":
        df_train_X, df_train_Y = df_train_X.as_matrix(), df_train_Y.as_matrix()
        df_test_X, df_test_Y = df_test_X.as_matrix(), df_test_Y.as_matrix()
    elif regression_type == "one_attr":
        df_train_X_attr, df_test_X_attr = df_train_X[attribute], df_test_X[attribute]
        df_train_X_bias, df_test_X_bias = df_train_X['bias'], df_test_X['bias']
        df_train_X = pd.concat([df_train_X_attr, df_train_X_bias],  axis=1)
        df_test_X = pd.concat([df_test_X_attr, df_test_X_bias],  axis=1)
        df_train_X, df_test_X = df_train_X.as_matrix(), df_test_X.as_matrix()
        df_test_Y, df_train_Y = df_test_Y.as_matrix(), df_train_Y.as_matrix()
    return df_train_X, df_train_Y, df_test_X, df_test_Y
    


def applyKFold(X, Y):
    kf = KFold(n_splits=5)
    # Holds the 5 fold cross validation splits
    training_folds = []
    for train_index, _ in kf.split(X):
        fold_X = []
        fold_Y = []
        for index in train_index:
            fold_X.append(X[index])
            fold_Y.append([Y[index]])
        training_folds.append((fold_X, fold_Y))
    return training_folds



