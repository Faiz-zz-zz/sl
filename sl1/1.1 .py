
get_ipython().magic('matplotlib inline')

import matplotlib
import numpy as np
import matplotlib.pyplot as plt



def feature_matrix(input_vector, basis_func, k):
    """
    Creates the feature matrix using the basis function
    eg., if basis function is  (1, x) and input is 10,
    this returns (1, 10)
    """
    matrix = []
    for elem in input_vector:
        matrix.append(
            [basis_func(elem, i) for i in range(k)]
        )
    return np.asarray(matrix)


def get_weights(phi, output):
    """
    Given the feature matrxi and output, returns the weights.
    """
    return np.linalg.lstsq(phi, output)[0]


# Question 1
data = [(1, 3),(2, 2),(3, 0),(4, 5)]
xs = [x for x, y in data]
ys = np.asarray([y for x, y in data])


weights = []
for k_val in range(1, 5):
    phi = feature_matrix(xs, lambda k, i: k ** i, k_val)
    weights.append(get_weights(phi, ys))


def predict_y(weights, xs):
    """
    Given the weight and input values, 
    returns the predicted values.
    Only works for basis function (1, x, ... x^k)
    """
    tot_arr = []
    for x in xs:
        total = 0
        for i, weight in enumerate(weights):
            total += (weight * (x ** i))
        tot_arr.append(total)
    return tot_arr


# Questions 1.1 1a
plt.axis([0, 6, -3, 6])
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xs, ys, 'ro')
line = [i/1000 for i in range(0, 6000)]
for i in range(1, 5):
    plt.plot(line, predict_y(weights[i - 1], line), label="k={}".format(i))

# Question 1.1 1b
for weight in weights:
    print(weight)


def mse(weight):
    """
    Given weights, returns the mse.
    """
    predicted_ys = predict_y(weight, xs)
    lse = sum([(ys[i] - predicted_ys[i]) ** 2 for i in range(len(ys))])
    return lse / len(ys)

# Question 1.1 1c
for i in range(1, 5):
    print("MSE for k = {} : {}".format(i, mse(weights[i - 1])))


# Question 2.1
from math import sin, pi
def g_func(x):
    return sin(2 * pi * x) ** 2


def create_random_data(n, sigma, mu):
    """
    For a given n, sigma and mu,
    returns random data with from a uniform distribution with noise
    drawn from a normal distribution with given sigma and mu.
    """
    S = []
    XS = []
    for _ in range(n):
        x = np.random.uniform()
        S.append(g_func(x) + np.random.normal(mu, sigma))
        XS.append(x)
    return XS, S


# Question 2.1 2ai
xs, sample_data = create_random_data(30, 0.07, 0)
sine = [i / 1000 for i in range(1, 1000)]
plt.axis([0, 1, 0, 1])
plt.xlabel("x")
plt.ylabel("y")
plt.plot(xs, sample_data, 'ro')
plt.plot(sine, [g_func(x) for x in sine])


weights = []
for k_val in [2, 5, 10, 14, 18]:
    phi = feature_matrix(xs, lambda k, i: k ** i, k_val)
    weights.append(get_weights(phi, sample_data))


# Question 2.1 2a(ii)
fig = matplotlib.pyplot.gcf()
fig.set_size_inches(18.5, 10.5)
plt.axis([-0.1, 1.1, -0.1, 1.1])
plt.plot(xs, sample_data, 'ro')
plt.plot(sine, [g_func(x) for x in sine])
for k, weight in enumerate(weights):
    plt.plot(sine, predict_y(weight, sine), label="k={}".format([2, 5, 10, 14, 18][k]))
    plt.legend()


def sin_mse(output, predicted):
    """
    Calculates the mse
    """
    total_error = 0
    for y, py in zip(output, predicted):
        total_error += (y - py) ** 2
    return total_error


weights = []
for k_val in range(1, 19):
    phi = feature_matrix(xs, lambda k, i: k ** i, k_val)
    weights.append(get_weights(phi, sample_data))


errors = []
ks = []
from math import log
for i, weight in enumerate(weights):
    predicted = predict_y(weight, xs)
    errors.append(log(sin_mse(output=sample_data, predicted=predicted)))
    ks.append(i + 1)


# Question 1.2 b
plt.plot(ks, errors)
plt.xlabel("k")
plt.ylabel("log(error)")


# Create 1000 test data points
test_x, test_y = create_random_data(1000, 0.07, 0)

# predict the output for the generated test points
errors, ks = [], []
for i, weight in enumerate(weights):
    predicted = predict_y(weight, test_x)
    errors.append(log(sin_mse(output=test_y, predicted=predicted)))
    ks.append(i + 1)


# Question 1.2c
plt.plot(ks, errors)
plt.xlabel("k")
plt.ylabel("log(error)")
plt.title("Test sample")


def predict_output(weights, xs, func):
    """
    For an given basis function and weights and input points, 
    returns predicted values.
    """
    tot_arr = []
    for x in xs:
        total = 0
        for i, weight in enumerate(weights):
            total += (weight * func(x, i))
        tot_arr.append(total)
    return tot_arr


def average_runs(func):
    """
    Given a basis function,
    function generates test and train set,
    perform linear regression with 1-18 dimensions and average the MSE over
    100 different independant runs.
    Returns the average test and train MSE.
    """
    train_err, test_err = [0] * 18, [0] * 18
    for _ in range(100):
        train_x, train_y = create_random_data(30, 0.07, 0)
        test_x, test_y = create_random_data(1000, 0.07, 0)
        weights = []
        for k_val in range(1, 19):
            phi = feature_matrix(train_x, func, k_val)
            weights.append(get_weights(phi, train_y))
        for i, weight in enumerate(weights):
            predicted = predict_output(weight, train_x, func)
            train_err[i] += (log(sin_mse(output=train_y, predicted=predicted)))
            predicted = predict_output(weight, test_x, func)
            test_err[i] += (log(sin_mse(output=test_y, predicted=predicted)))
    return list(map(lambda k: k / 100, train_err)), list(map(lambda k: k / 100, test_err))


train, test = average_runs(lambda k, i: k ** i)



plt.plot([i for i in range(1, 19)], train)
plt.title("Train error")
plt.xlabel("k")
plt.ylabel("log(error)")


plt.plot([i for i in range(1, 19)], test)
plt.title("Test error")
plt.xlabel("k")
plt.ylabel("log(error)")


# do 100 runs with (...,sin(i(pi)k)) as basis function
train, test = average_runs(lambda k, i: sin((i + 1) * pi * (k)))

# Question 3
plt.plot([i for i in range(1, 19)], train)
plt.plot([i for i in range(1, 19)], test)

weights = []

for k_val in range(1, 19):
    phi = feature_matrix(xs, lambda k, i: sin(i * pi * k), k_val)
    weights.append(get_weights(phi, sample_data))


def predict_function(xs, weights, func):
    predicted = []
    for x in xs:
        total = 0
        for i, w in enumerate(weights):
            total += w * func(x, i)
        predicted.append(total)
    return predicted


train_err, test_err = [0] * 18, [0] * 18
test_x, test_y = create_random_data(1000, 0.07, 0)
func = lambda k, i: sin(i * pi * k)
for i, weight in enumerate(weights):
    predicted = predict_output(weight, xs, func)
    train_err[i] += (log(sin_mse(output=sample_data, predicted=predicted)))
    predicted = predict_output(weight, test_x, func)
    test_err[i] += (log(sin_mse(output=test_y, predicted=predicted)))


plt.plot([i + 1 for i in range(18)], train_err)
plt.title("Train error")
plt.xlabel("k")
plt.ylabel("log(error)")


plt.plot([i + 1 for i in range(18)], test_err)
plt.title("Test error")
plt.xlabel("k")
plt.ylabel("log(error)")

# do 100 runs with (...,sin(i(pi)k)) as basis function
train_err, test_err = average_runs(lambda k, i: (sin(i * k * pi)))

plt.plot([i + 1 for i in range(18)], train_err)
plt.title("Train error over 100 runs")
plt.xlabel("k")
plt.ylabel("log(error)")

plt.plot([i + 1 for i in range(18)], test_err)
plt.title("Test error over 100 runs")
plt.xlabel("k")
plt.ylabel("log(error)")


