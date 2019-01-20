import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def addIntercept(X):
    int = np.ones((X.shape[0], 1))
    return np.concatenate((int, X), axis=1)

def sigmoid(X, theta):
    z = np.dot(X, theta.transpose())
    return 1 / (1 + np.exp(-z)) #sigmoid z

def costFunction(h, y):
    inner = -y * np.log(h) - (1 - y) * np.log(1 - h)
    cost = np.sum(inner) / len(y)
    return cost

def gradientDescent(X, y, theta, maxIterations, alpha, stop, regParameter):
    m = len(X)
    J = []
    it = 0

    h = sigmoid(X, theta)
    oldCost =costFunction(h, y)
    J.append(oldCost)
    it += 1

    for i in range(maxIterations):
        gradient = np.dot(X.T, (h - y)) * (alpha / m)

        if regParameter == 'NA':
            theta = theta - gradient.transpose()
        else:
            theta[0, 0] = theta[0, 0] - gradient.transpose()[0,0]
            theta[0, 1:] = theta[0, 1:] * (1 - alpha * (regParameter / m)) - gradient.transpose()[0,1:]

        h = sigmoid(X, theta)
        cost = costFunction(h, y)
        J.append(cost)
        it +=1

        if abs(cost - oldCost) <= stop:
            return theta, J, it

        else:
            oldCost = cost

    return theta, J, it

def prediction(X_test, theta):
    prob = sigmoid(X_test, theta)
    return prob.round()

# ==================== Logistic Regression ====================
# parameters
num_iterations = 10000
alpha = 0.5
delta_J_threshold = 0.00001

# create column names
col_names = ['ID', 'Class']
for i in range(1,31):
    col_names.append('Att. {}'.format(i))

# load data
data = pd.read_csv('/Users/whitneyware/Classes/Machince Learning/wdbc.data.txt', names = col_names, index_col=False)
data['Class'] = data['Class'].replace('B', 0)
data['Class'] = data['Class'].replace('M', 1)
X = data.values[:,2:]
Y = data['Class']
Y = Y.values.reshape((len(Y), 1))
m = X.shape[0]
n = X.shape[1]

# pre-treatment
X_pretreated = np.zeros((m, n))
for i in range(n):
    cur_col = X[:, i]

    col_min = min(cur_col) * np.ones(m)
    col_range = (max(cur_col) - min(cur_col)) * np.ones(m)

    X_pretreated[:, i] = (cur_col - col_min) / col_range

# add intercept and split
X_pretreated = addIntercept(X_pretreated)

# test prediction using loocv
y_test_vals = []
predicted_y = []
sklearn_predict = []
iter = 10
for i in range(iter):
    X_train, X_test, Y_train, Y_test = train_test_split(X_pretreated, Y, test_size=1)
    theta = np.zeros([1, X_train.shape[1]])

    t, J, i = gradientDescent(X_train, Y_train, theta, num_iterations, alpha, delta_J_threshold, regParameter='NA')
    pred = prediction(X_test, t)
    y_test_vals.append(Y_test[0][0])
    predicted_y.append(int(pred[0][0]))

    #sklearn
    sk = LogisticRegression(random_state=0, fit_intercept=False)
    sk.fit(X_train, Y_train)
    sk1_predict = sk.predict(X_test)
    sklearn_predict.append(sk1_predict[0])

# print Y test, predicted Y, and sklearn y
print('Y Test Values: ')
print(y_test_vals)
print('Logistic Regression Class Predicted Y Values: ')
print(predicted_y)
print('Sklearn Predicted Y Values: ')
print(sklearn_predict)

#error rate
error_act = sum(i != j for i, j in zip(y_test_vals, predicted_y))
error_sk = sum(i != j for i, j in zip(sklearn_predict, y_test_vals))
print('Percent Error for My Logistic Regression: ' + str(error_act/iter * 100) + '%')
print('Percent Error for Sklearn: ' + str(error_sk/iter * 100) + '%')
