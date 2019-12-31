from sklearn import datasets
import numpy as np


# sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# normalization
# return normalized data
def dataProcess(data):
    columns = data.shape[1]
    for i in range(columns):
        max = np.max(data[:, i])
        min = np.min(data[:, i])
        data[:, i] = (data[:, i] - min) / (max - min)
    return data


# predict value
# input
# data: training data, m * 10
# theta: array of parameters, 1 * 10
# return: prediction of each instances, m * 1
def predictValue(data, theta):
    return sigmoid(np.dot(data, theta.reshape(-1, 1)))


# cost function
# input
# m: num of instances;
# theta: array of parameters, 1 * 10
# target: 1 * m
# return cost
def costFunction(data, theta, target):
    m = data.shape[0]

    cost = 1 / (2 * m) * (np.square(predictValue(data, theta) - target).sum())
    return cost


def costFunctionRegular(data, theta, target, lam):
    m = data.shape[0]

    # regularization
    cost = 1 / (2 * m) * (np.square(predictValue(data, theta) - target).sum() + (theta ** 2).sum())
    return cost


# gradient descent
# input target: labels, 1 * m
def gradientDescent(data, theta, target, learningRate):
    cost = costFunction(data, theta, target)
    cost1 = 0
    m = data.shape[0]
    while abs(cost - cost1) > 1e-3:
        cost1 = cost
        error = predictValue(data, theta) - target.reshape(-1, 1)
        theta = theta - learningRate * (1 / m) * np.dot(error.T, data)
        cost = costFunction(data, theta, target)

    return theta, cost


def gradientDescentRegular(data, theta, target, learningRate, lam):
    cost = costFunctionRegular(data, theta, target, lam)
    cost1 = 0
    m = data.shape[0]

    while abs(cost - cost1) > 1e-4:
        cost1 = cost
        error = predictValue(data, theta) - target.reshape(-1, 1)
        theta = theta - learningRate * (1 / m) * (np.dot(error.T, data) + lam / m * theta)
        cost = costFunctionRegular(data, theta, target, lam)

    return theta, cost


# compute the MSE error in the test data
def accuracy(theta, data, target):
    m = data.shape[0]
    predict = predictValue(data, theta)
    correctNum = 0
    for i in range(m):
        if (predict[i] >= 0.5) == target[i]:
            correctNum += 1

    return correctNum / m


if __name__ == '__main__':
    cancer = datasets.load_breast_cancer()

    data = cancer.data
    target = cancer.target
    data = dataProcess(data)

    # num of training data
    m = data.shape[0]

    # split test data and tran data
    trainData = data[:int(0.8 * m)]
    trainLabel = target[:int(0.8 * m)]

    testData = data[int(0.8 * m):]
    testLabel = target[int(0.8 * m):]

    # initialize theta randomly
    theta = np.ones(data.shape[1])

    learningRate = 0.01

    # theta, cost = gradientDescent(trainData, theta, trainLabel, learningRate)
    #
    thetaReg1, costReg1 = gradientDescentRegular(trainData, theta, trainLabel, learningRate, 0.3)
    #
    # thetaReg5, costReg5 = gradientDescentRegular(trainData, theta, trainLabel, learningRate, 5)

    # thetaReg10, costReg10 = gradientDescentRegular(trainData, theta, trainLabel, learningRate, 10)
    # print(accuracy(theta, testData, testLabel), cost)
    print(accuracy(thetaReg1, testData, testLabel), costReg1)
    # print(accuracy(thetaReg5, testData, testLabel), costReg5)
    # print(accuracy(thetaReg10, testData, testLabel), costReg10)
    # 0.7105263157894737
    # 56.73673404773269
    # 0.9035087719298246
    # 87.97067398178062
    # 0.9035087719298246
    # 87.81638421093201
    # 0.9035087719298246
    # 87.62752113599709
