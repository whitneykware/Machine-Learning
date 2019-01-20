import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import train_test_split

class TwoClassLDA():
    def __init__(self, X, Y, numClasses, numFeatures):
        self.mean_vectors = []
        self.X = X
        self.Y = Y
        self.numClasses = numClasses
        self.numFeatures = numFeatures

    def fitLDA(self):
        # class means
        for i in range(self.numClasses):
            self.mean_vectors.append(np.mean(self.X[self.Y == i], axis = 0))

        # within class scatter matrix
        self.Swithin = np.zeros((self.numFeatures, self.numFeatures))
        for c in range(self.numClasses):
            class_scatter = np.zeros((self.numFeatures, self.numFeatures))
            for row in X[Y==c]:
                row = row.reshape(self.numFeatures, 1)
                m = self.mean_vectors[c].reshape(self.numFeatures, 1)
                class_scatter += (row - m).dot((row - m).T)
            self.Swithin += class_scatter

        # total mean
        self.total_mean = np.mean(X, axis = 0)

        # between class scatter
        self.Sbetween = np.zeros((self.numFeatures, self.numFeatures))
        for c in range(self.numClasses):
            num = self.X[self.Y==c].shape[0]
            m = self.mean_vectors[c].reshape(self.numFeatures, 1)
            self.total_mean = self.total_mean.reshape(self.numFeatures, 1)
            self.Sbetween += num * (m - self.total_mean).dot((m - self.total_mean).T)

        # get eig values and vectors
        self.eigValues, self.eigVectors = np.linalg.eig(np.linalg.inv(self.Swithin).dot(self.Sbetween))
        self.eigList = [(np.abs(self.eigValues[i]), self.eigVectors[:,i]) for i in range(len(self.eigValues))]

        # sort the eig pairs in descending order
        self.eigList = sorted(self.eigList, key=lambda k: k[0], reverse=True)

        # W (d x k -1)
        self.W = self.eigList[0][1]
        #self.W = np.array([self.eigList[i][1] for i in range(self.numFeatures)])

    def getW(self):
        return self.W

    def transform(self):
        X_transformed = self.X.dot(self.W)
        return X_transformed

    def predict(self, X_test):
        # get threshold by getting the mean of the class projection means
        mu0 = np.dot(self.W.reshape(1, self.numFeatures), self.mean_vectors[0])
        mu1 = np.dot(self.W.reshape(1, self.numFeatures), self.mean_vectors[1])
        self.w0 = sum(mu0, mu1) / 2

        # use threshold to assign to class
        y_predict = []
        for i in X_test:
            Xpro = np.dot(self.W, i.T)
            if Xpro >= self.w0:
                y_predict.append(1)
            else:
                y_predict.append(0)
        return y_predict

# ========================= LDA =========================
# create column names
col_names = ['ID', 'Class']
for i in range(1,31):
    col_names.append(i)

# load data
data = pd.read_csv('/Users/whitneyware/Classes/Machince Learning/wdbc.data.txt', names = col_names, index_col=False)
data['Class'] = data['Class'].replace('B', 0)
data['Class'] = data['Class'].replace('M', 1)
X = data[col_names[2:]].values
Y = data['Class'].values
m = X.shape[0]
n = X.shape[1]

# pre-treatment
X_pretreated = np.zeros((m, n))
for i in range(n):
    cur_col = X[:, i]

    col_min = min(cur_col) * np.ones(m)
    col_range = (max(cur_col) - min(cur_col)) * np.ones(m)

    X_pretreated[:, i] = (cur_col - col_min) / col_range

# LDA test prediction using loocv
y_test_vals = []
predicted_y = []
sklearn_predict = []
iter = 100
for i in range(iter):
    X_train, X_test, Y_train, Y_test = train_test_split(X_pretreated, Y, test_size=1)
    lda = TwoClassLDA(X_train, Y_train, numClasses = 2, numFeatures = 30)
    lda.fitLDA()
    transform_X = lda.transform()
    pred_lda = lda.predict(X_test)
    y_test_vals.append(Y_test[0])
    predicted_y.append(pred_lda[0])

    # sklearn
    model = LDA(n_components=2)
    skl_lda = model.fit(X_train, Y_train)
    sk_transform_X = model.transform(X)
    pred = skl_lda.predict(X_test)
    sklearn_predict.append(pred[0])

# print Y test, predicted Y, and sklearn y
print('Y Test Values: ')
print(y_test_vals)
print('LDA CLass Predicted Y Values: ')
print(predicted_y)
print('Sklearn Predicted Y Values: ')
print(sklearn_predict)

#error rate
error_act = sum(i != j for i, j in zip(y_test_vals, predicted_y))
error_sk = sum(i != j for i, j in zip(sklearn_predict, y_test_vals))
print('Percent Error for LDA Class: ' + str(error_act/iter * 100) + '%')
print('Percent Error for Sklearn: ' + str(error_sk/iter * 100) + '%')
