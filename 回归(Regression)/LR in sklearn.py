import numpy as np
import sklearn
from sklearn import linear_model, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

cancer = datasets.load_breast_cancer()
# min-max normalization
minmaxScaler = sklearn.preprocessing.MinMaxScaler()
data = minmaxScaler.fit_transform(cancer.data)

# split into train and test data
trainData, testData, trainLabel, testLabel = train_test_split(
    data, cancer.target, test_size=0.2)

cls = linear_model.LogisticRegression\
    (solver='liblinear' ,penalty='l2', C=10)
cls.fit(trainData, trainLabel)
prediction = cls.predict(testData)
a = cls.score(testData, testLabel)
print(a)









