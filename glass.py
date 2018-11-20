# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np


def readFile():
    glass = pd.read_csv("glass.csv")  # read file
    glass.info()  # frame info
    np.unique(glass.Type)  # get labels
    return glass


from sklearn.model_selection import KFold


def initKFold():
    kf = KFold(n_splits=10, shuffle=True)
    return kf


from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import math


def trainning():
    avg_accuracy = 0
    j = 0
    model = GaussianNB()
    for train_index, test_index in kf.split(X):
        j += 1
        print("= Lan lap", str(j), "=")
        print("Train:\n", train_index)
        print("Test:\n", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    #    model = GaussianNB()
        model.fit(X_train, y_train)

        reality = y_test.values
        predict = model.predict(X_test)
        common_value = np.array([p for p in predict if p in reality])

        subtracted = predict - reality
        count_zero = 0
        for i in subtracted:
            if i == 0:
                count_zero += 1

        print("reality: \n", reality)
        print("predict: \n", predict)
        print("common_value v1: \n", subtracted)
        print("count zero: \n", np.count_nonzero(
            subtracted == 0), "/", len(subtracted))

        y_pred = predict
        accuracy = accuracy_score(y_test, y_pred)*100
        if math.isnan(accuracy) == False:
            avg_accuracy += accuracy

        print("Accuracy at", j, ":", accuracy)
        print("Do chinh xac trung binh at", j, ":", avg_accuracy/j)

    #    labels = np.unique(glass.Type)
    #    matrix = confusion_matrix(reality, predict, labels=labels)
    #
    #    leni = len(matrix)
    #    accuracy_avg = 0;
    #    for i in range(0,leni):
    #        accuracy = float(matrix[i][i])/(np.sum(matrix[i]))*100
    #        accuracy_avg += accuracy
    #        print('Do chinh xac lop '+ str(i) + ": " + str(accuracy) + "%")
    #
    #    print("X_test:", X_test)
        print("==================")
    print("Do chinh xac trung binh:", avg_accuracy/10)
    return model


# predict
def testModel(inputSample=np.array(
        [[1.51556, 13.87, 0, 2.54, 73.23, 0.14, 9.41, 0.81, 0.01]])):
    resultSample = model.predict(inputSample)
    print("result sample:", resultSample)
    return resultSample


import json
from flask import Flask, jsonify, request
from flask_restful import reqparse, abort, Api, Resource
app = Flask(__name__)
api = Api(app)

parser = reqparse.RequestParser()

'''
Test by post man
    {
        "RI":  1.51316,
        "Na":  13.02,
        "Mg":  0,
        "Al":  3.04,
        "Si":  70.48,
        "K":  6.21,
        "Ca":  6.96,
        "Ba":  0,
        "Fe":  0
    }
'''

class HelloWorld(Resource):
    def get(self):
        result = testModel()
        return {
            'hello': 'world',
            'input': json.dumps(
                [1.51556, 13.87, 0, 2.54, 73.23, 0.14, 9.41, 0.81, 0.01]),
            'result': str(result)
        }
    def post(sefl):
        content = request.json
        inputSample = np.array([list(content.values())])
        result = int(testModel(inputSample)[0])
        print('requestBody:', inputSample)
        print('result:', result)
        print("type: ", result)
        return {
            'Type' : result
        }


api.add_resource(HelloWorld, '/')


glass = readFile()
X = glass.iloc[:, 0:9]
y = glass.iloc[:, 9]
kf = initKFold()
model = trainning()
# testModel()


def hello():
    return jsonify({
        "hello": 'world',
        "do chinh xac": avg_accuracy/10
    })


if __name__ == '__main__':
    app.run(debug=True)
