import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def costFunction(X, y, theta):
    residuals = (X @ theta.T) - y
    cost = np.sum(residuals ** 2) / (2 * len(X))
    return cost

def gradientDescent(X, y, theta, maxIterations, alpha, stop):
    m = len(X)
    J = []
    it = 0

    oldCost = costFunction(X, y, theta)
    J.append(oldCost)
    it += 1

    for i in range(maxIterations):
        residuals = (X @ theta.T) - y
        gradient = (alpha / m) * np.sum(X * residuals, axis=0)
        theta = theta - gradient
        cost = costFunction(X, y, theta)
        J.append(cost)
        it +=1

        if abs(cost - oldCost) <= stop:
            return theta, J, it

        else:
            oldCost = cost

    return theta, J, it

def predict(X, theta):
    y_hat = X @ theta.T
    return y_hat

# read in data
home_data = pd.read_csv("/Users/whitneyware/Classes/Machince Learning/home_price.csv")

# z-score normalization
cols = list(home_data.columns)
for col in cols:
    home_data[col] = (home_data[col] - home_data[col].mean())/home_data[col].std(ddof=0)

# ===== LR - price vs size =====
X_homeSize = home_data.iloc[:,0:1]
X = np.concatenate((np.ones([X_homeSize.shape[0],1]), X_homeSize), axis=1)
y = np.array(home_data.iloc[:,2:3])
n = X.shape[1]
theta = np.zeros([1, n])
print(type(X))

# alpha = 0.01
t1, J1, i1 = gradientDescent(X, y, theta, maxIterations = 10000, alpha = 0.01, stop = 0.001)
fig1, ax = plt.subplots()
ax.plot(np.arange(i1), J1, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations for alpha = 0.01')
print(t1)

# alpha = 0.1
t2, J2, i2 = gradientDescent(X, y, theta, maxIterations = 10000, alpha = 0.1, stop = 0.001)
fig2, ax = plt.subplots()
ax.plot(np.arange(i2), J2, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations for alpha = 0.1')

# alpha = 1
t3, J3, i3 = gradientDescent(X, y, theta, maxIterations= 10000, alpha = 1.0 , stop = 0.001)
fig3, ax = plt.subplots()
ax.plot(np.arange(i3), J3, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations for alpha = 1')

# alpha = 0.001
t4, J4, i4 = gradientDescent(X, y, theta, maxIterations= 10000, alpha=0.001, stop = 0.001)
fig4, ax = plt.subplots()
ax.plot(np.arange(i4), J4, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations for alpha = 0.001')

# alpha = 0.001, stop = 0.00001
t5, J5, i5 = gradientDescent(X, y, theta, maxIterations= 10000, alpha=0.001, stop = 0.00001)
fig5, ax = plt.subplots()
ax.plot(np.arange(i5), J5, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Cost vs. Iterations for alpha = 0.001 with Stop Criterion = 0.00001')

# linear relationship for alpha = 0.001
predicted = predict(X, t4)
fig6, ax = plt.subplots()
ax.scatter(X_homeSize, y, color='black', marker="o", linewidth=2)
ax.plot(X_homeSize, predicted, color='blue', linewidth=2, label = "predicted")
ax.set_xlabel('Home Size')
ax.set_ylabel('Home Price')
ax.set_title("Gradient Descent Based Linear Regression, a = 0.001, stop = 0.001")
ax.legend(loc='lower right', fontsize=9)

predicted = predict(X, t5)
fig7, ax = plt.subplots()
ax.scatter(X_homeSize, y, color='black', marker="o", linewidth=2)
ax.plot(X_homeSize, predicted, color='blue', linewidth=2, label = "predicted")
ax.set_xlabel('Home Size')
ax.set_ylabel('Home Price')
ax.set_title("Gradient Descent Based Linear Regression, a = 0.01, stop = 0.00001")
ax.legend(loc='lower right', fontsize=9)

# plot home price vs home size with linear relationship from GD for learning rate = 0.01
predicted = predict(X, t1)
fig8, ax = plt.subplots()
ax.scatter(X_homeSize, y, color='black', marker="o", linewidth=2)
ax.plot(X_homeSize, predicted, color='blue', linewidth=2, label = "predicted")
ax.set_xlabel('Home Size')
ax.set_ylabel('Home Price')
ax.set_title("Gradient Descent Based Linear Regression, a = 0.01")
ax.legend(loc='lower right', fontsize=9)
#plt.show()

# ===== MLR - price vs size and bedrooms =====
X_mv = home_data.iloc[:,0:2]
X_mv = np.concatenate((np.ones([X_mv.shape[0], 1]), X_mv), axis=1)
y_mv = np.array(home_data.iloc[:,2:3])
n_mv = X_mv.shape[1]
theta_mv = np.zeros([1, n_mv])

t_mv, J_mv, i_mv = gradientDescent(X_mv, y_mv, theta_mv, maxIterations= 10000, alpha = 0.01, stop = 0.001)
predicted_mv = predict(X_mv, t_mv)
#print(t_mv)
