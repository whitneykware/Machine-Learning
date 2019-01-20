import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import seaborn as sns
sns.set(style='white')
sns.set(style='whitegrid', color_codes=True)

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

# ===== parameters=====
num_iterations = 10000
alpha = 0.1
delta_J_threshold = 0.00001

# =====load data =====
iris = load_iris()
II = (iris.target != 0)

X = iris.data[II, 2:4]
Y = iris.target[II]
Y = Y.reshape((len(Y), 1))

II_1 = np.where(Y == 1)
II_2 = np.where(Y == 2)

Y[II_1] = 0.
Y[II_2] = 1.0

m = X.shape[0]
n = X.shape[1]

# ===== pre-treatment of variables ======
X_pretreated = np.zeros((m, n))
for i in range(n):
    cur_col = X[:, i]

    col_min = min(cur_col) * np.ones(m)
    col_range = (max(cur_col) - min(cur_col)) * np.ones(m)

    X_pretreated[:, i] = (cur_col - col_min) / col_range

# ===== add intercept and split =====
X_pretreated = addIntercept(X_pretreated)
X_train, X_test, Y_train, Y_test = train_test_split(X_pretreated, Y, test_size=1, random_state=0)
theta = np.zeros([1, X_train.shape[1]])

# ========== Logistic Regression ==========
LG_results = []

# ===== No regularization =====
t1, J1, i1 = gradientDescent(X_train, Y_train, theta, num_iterations, alpha, delta_J_threshold, regParameter='NA')
pred1 = prediction(X_test, t1)
fig1, ax = plt.subplots()
ax.plot(np.arange(i1), J1, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations, a = 0.1, delta J = 1e-5 \n Without Regularization')
LG_results.append(pred1[0][0])

# ===== Regularization, lambda = 0.5 =====
t2, J2, i2= gradientDescent(X_train, Y_train, theta, num_iterations, alpha, delta_J_threshold, regParameter=0.5)
pred2 = prediction(X_test, t2)
fig2, ax = plt.subplots()
ax.plot(np.arange(i2), J2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations, a = 0.1, delta J = 1e-5 \n With Regularization, lambda = 0.5')
LG_results.append(pred2[0][0])

# ===== Regularization, lambda = 1.0 =====
t3, J3, i3 = gradientDescent(X_train, Y_train, theta, num_iterations, alpha, delta_J_threshold, regParameter=1.0)
pred3 = prediction(X_test, t3)
fig3, ax = plt.subplots()
ax.plot(np.arange(i3), J3, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations, a = 0.1, delta J = 1e-5 \n With Regularization, lambda = 1.0')
LG_results.append(pred3[0][0])

# ===== Regularization, lambda = 1.5 =====
t4, J4, i4 = gradientDescent(X_train, Y_train, theta, num_iterations, alpha, delta_J_threshold,regParameter=1.5)
pred4 = prediction(X_test, t4)
fig4, ax = plt.subplots()
ax.plot(np.arange(i4), J4, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations, a = 0.1, delta J = 1e-5 \n With Regularization, lambda = 1.5')
LG_results.append(pred4[0][0])

plt.show()

# ========== SCIKIT LEARN ==========
sk_results = []

def costFunctionWithReg(coef, x, y, regParameter):
    inner = -y * np.log(sigmoid(x, coef)) - (1 - y) * np.log(1 - sigmoid(x, coef))
    regCost = (regParameter*np.sum(theta[1:]**2))/(2*m)
    cost = np.sum(inner)/ m + regCost
    return cost

# ===== sklearn, no regularization =====
sk1 = LogisticRegression(random_state=0, fit_intercept=False)
sk1.fit(X_train, Y_train)
sk1_predict = sk1.predict(X_test)
sk_results.append(sk1_predict[0])
sk1_cost = costFunction(sigmoid(X_train, sk1.coef_), Y_train)
print("SciKit-Learn without regularization: \ncoefficients: {}\n"
      "prediction: {}\nfinal cost: {}\n".format(sk1.coef_, sk1_predict, sk1_cost))

# ===== sklearn, lambda = 0.5 =====
sk2 = LogisticRegression(random_state=0, fit_intercept=False, C=1/0.5)
sk2.fit(X_train, Y_train)
sk2_predict = sk2.predict(X_test)
sk_results.append(sk2_predict[0])
sk2_cost = costFunctionWithReg(sk2.coef_, X_train, Y_train, regParameter=0.5)
print("SciKit-Learn with regularization, lambda = 0.5: \ncoefficients: {}\n"
      "prediction: {}\nfinal cost: {}\n".format(sk2.coef_, sk2_predict, sk2_cost))

# ===== sklearn, lambda = 1.0 =====
sk3 = LogisticRegression(random_state=0, fit_intercept=False, C=1/1.0)
sk3.fit(X_train, Y_train)
sk3_predict = sk3.predict(X_test)
sk_results.append(sk3_predict[0])
sk3_cost = costFunctionWithReg(sk3.coef_, X_train, Y_train, regParameter=1.0)
print("SciKit-Learn with regularization, lambda = 1.0: \ncoefficients: {}\n"
      "prediction: {}\nfinal cost: {}\n".format(sk3.coef_, sk3_predict, sk3_cost))

# ===== sklearn, lmabda = 1.5 =====
sk4 = LogisticRegression(random_state=0, fit_intercept=False, C=1/1.5)
sk4.fit(X_train, Y_train)
sk4_predict = sk4.predict(X_test)
sk_results.append(sk4_predict[0])
sk4_cost = costFunctionWithReg(sk4.coef_, X_train, Y_train, regParameter=1.5)
print("SciKit-Learn with regularization, lambda = 1.5: \ncoefficients: {}\n"
      "prediction: {}\nfinal cost: {}\n".format(sk4.coef_, sk4_predict, sk4_cost))


print("Actual y test: {}\n".format(Y_test[0][0]))
print("Logistic Regression Predictions: ")
print(LG_results)
print()
print("SciKit-Learn Predictions: ")
print(sk_results)
