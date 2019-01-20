import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

class TwoHiddenLayerNeuralNetwork:
    def __init__(self, inputSize, numOutputs, numHiddenUnits, learningRate):
        self.n = inputSize
        self.units = numHiddenUnits
        self.numOutputs = numOutputs
        self.alpha = learningRate
        self.b1 = np.random.rand(1)
        self.b2 = np.random.rand(1)
        self.b3 = np.random.rand(1)
        self.w1 = np.random.rand(self.n, self.units) # 30 x 10
        self.w2 = np.random.rand(self.units, self.units) # 10 x 10
        self.w3 = np.random.rand(self.units, self.numOutputs) # 10 x 1

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def sigmoidDer(self, z):
        return z * (1.0 - z)

    def forwardProp(self, X):
        z1 = np.dot(X, self.w1) + self.b1
        self.layer1 = self.sigmoid(z1)
        z2 = np.dot(self.layer1, self.w2) + self.b2
        self.layer2 = self.sigmoid(z2)
        z3 = np.dot(self.layer2, self.w3) + self.b3
        self.output = self.sigmoid(z3)

    def backProp(self, X_train, Y_train):
        outputError = Y_train - self.output
        outputDelta = outputError * self.sigmoidDer(self.output)

        layer2_Error = np.dot(outputDelta, self.w3.T)
        layer2_Delta = layer2_Error * self.sigmoidDer(self.layer2)

        layer1_Error = np.dot(layer2_Delta, self.w2.T)
        layer1_Delta = layer1_Error * self.sigmoidDer(self.layer1)

        delta_w1 = np.dot(X_train.T, layer1_Delta)
        delta_w2 = np.dot(self.layer1.T, layer2_Delta)
        delta_w3 = np.dot(self.layer2.T, outputDelta)

        self.w1 += self.alpha * delta_w1
        self.w2 += self.alpha * delta_w2
        self.w3 += self.alpha * delta_w3

        self.b1 = 1 / Y_train.shape[0] * np.sum(layer1_Delta, axis=0)
        self.b2 = 1 / Y_train.shape[0] * np.sum(layer2_Delta, axis=0)
        self.b3 = 1 / Y_train.shape[0] * np.sum(outputDelta, axis=0)

    def train(self, X_train, Y_train):
        self.forwardProp(X_train)
        self.backProp(X_train, Y_train)

    def predict(self, X_test):
        y_pred = []
        z1 = np.dot(X_test, self.w1) + self.b1
        layer1 = self.sigmoid(z1)
        z2 = np.dot(layer1, self.w2) + self.b2
        layer2 = self.sigmoid(z2)
        z3 = np.dot(layer2, self.w3) + self.b3
        out = self.sigmoid(z3)
        if out >= 0.5:
            y_pred.append(1)
        else:
            y_pred.append(0)
        return out, y_pred

# ==================== Neural Network ====================
# create column names
col_names = ['ID', 'Class']
for i in range(1,31):
    col_names.append('Att. {}'.format(i))

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

# test prediction using loocv
y_test_vals = []
predicted_y = []
sklearn_predict = []
outList = []
iter = 10

for i in range(iter):
    X_train, X_test, Y_train, Y_test = train_test_split(X_pretreated, Y, test_size=1)
    Y_train = Y_train.reshape(len(Y_train), 1)

    nn = TwoHiddenLayerNeuralNetwork(inputSize=30, numOutputs=1, numHiddenUnits=5, learningRate=0.01)
    for x in range(1000): # trains the NN 1,000 times
        nn.train(X_train, Y_train)
    out, pred = nn.predict(X_test)
    y_test_vals.append(Y_test[0])
    predicted_y.append(pred[0])
    outList.append(out[0][0])

    #sklearn
    MLP = MLPClassifier(solver='lbfgs', activation='logistic', alpha=0.01, hidden_layer_sizes=(5,5))
    MLP.fit(X_train, Y_train)
    skpred = MLP.predict(X_test)
    sklearn_predict.append(skpred[0])

# print Y test, predicted Y, and error
print('Y test Values: ')
print(y_test_vals)
print('Predicted Y Values: ')
print(predicted_y)
print('Sklearn Predicted Y Values: ')
print(sklearn_predict)
print('Predicted output: ')
print(outList)

error = sum(i != j for i, j in zip(y_test_vals, predicted_y))
error_sk = sum(i != j for i, j in zip(sklearn_predict, y_test_vals))
print('Percent Error for Neural Network: ' + str(error/iter * 100) + '%')
print('Percent Error for Sklearn: ' + str(error_sk/iter * 100) + '%')