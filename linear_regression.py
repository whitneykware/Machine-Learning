import os, sys
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import datasets

diabetes = datasets.load_diabetes()
x_train, x_test, y_train, y_test = train_test_split(diabetes.data[:,2], diabetes.target, test_size=20, random_state=0)


#======lInear regression using class======

class LinearRegression:

    def train(self, x_train, y_train):
        self.x_train = np.array(x_train)
        self.y_train = np.array(y_train)

        self.x_bar = np.mean(self.x_train)
        self.y_bar = np.mean(self.y_train)

        self.S_yx = np.sum((self.y_train - self.y_bar) * (self.x_train - self.x_bar))
        self.S_xx = np.sum((self.x_train - self.x_bar) ** 2)

        self.beta_1_hat = self.S_yx / self.S_xx
        self.beta_0_hat = self.y_bar - self.beta_1_hat * self.x_bar


    def predict(self, x):
        self.x = np.array(x)
        self.pred = self.beta_0_hat + self.beta_1_hat * self.x
        return self.pred


diaLM = LinearRegression()
diaLM.train(x_train, y_train)
y_predicted = diaLM.predict(x_test)

# ======linear regression using sci-kit learn======

x_train = x_train.reshape((len(x_train), 1))
x_test = x_test.reshape((len(x_test), 1))
lm_sklearn= linear_model.LinearRegression()
lm_sklearn.fit(x_train, y_train)
y_hat = lm_sklearn.predict(x_test)

# ======plots======

n = len(x_test)
x_bar = np.mean(x_test)
y_bar = np.mean(y_test)

# class plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_test, y_test, color='red', marker="o", linewidth=2, label = "test set")
ax.plot(x_test, y_predicted, color='blue', marker="D", linewidth=2, label = "LR class predicted")
ax.plot(x_test, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=2)
ax.plot([x_bar, x_bar], [np.min(y_test), np.max(y_test)], color='black', linestyle=':', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Linear Regression Using Class")
ax.legend(loc='lower right', fontsize=9)
fig.show()

# sklearn plot
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_test, y_test, color='red', marker="o", linewidth=2, label = "test set")
ax.plot(x_test, y_hat, color='blue', marker="D", linewidth=2, label = "predicted")
ax.plot(x_test, np.ones(n)*y_bar, color='black', linestyle=':', linewidth=2)
ax.plot([x_bar, x_bar], [np.min(y_test), np.max(y_test)], color='black', linestyle=':', linewidth=2)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title("Linear Regression Using Sci-kit Learn")
ax.legend(loc='lower right', fontsize=9)
fig.show()
