

import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
import pandas

def one_hot(dataFrame, feature):

    onehot = pandas.get_dummies(dataFrame[feature])
    dataFrame = dataFrame.drop(feature, axis=1)
    dataFrame = dataFrame.join(onehot)

    return dataFrame

def stdNormalize(dataFrame, columns):

    std_scale = StandardScaler().fit(dataFrame[columns])
    dataFrame_std = std_scale.transform(dataFrame[columns])
    dataFrame = dataFrame[columns] = dataFrame_std
    return dataFrame

def getData():

    dataFrame = pandas.read_csv("https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/ann_logistic_extra/ecommerce_data.csv")

    #shuffle
    dataFrame = shuffle(dataFrame)
    Y = dataFrame.pop('user_action')

    #one_hot_encode
    dataFrame = one_hot(dataFrame, 'time_of_day')

    #split
    Xtrain = dataFrame[:-100]
    Ytrain = Y[:-100]

    Xtest = dataFrame[-100:]
    Ytest = Y[-100:]

    #normalize

    columns = ["n_products_viewed", "visit_duration"]
    Xtrain = stdNormalize(Xtrain, columns)
    Xtest = stdNormalize(Xtest, columns)


    return Xtrain, Ytrain, Xtest, Ytest



def get_binary_data():
    Xtrain, Ytrain, Xtest, Ytest = getData()
    return Xtrain[Ytrain < 1], Ytrain[Ytrain < 1], Xtest[Ytest < 1], Ytest[Ytest < 1]









