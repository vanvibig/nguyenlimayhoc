# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
# 1a
glass = pd.read_csv("glass.csv")
# 1b
len(glass)
# so phan tu: 1599
glass.info()
labels = np.unique(glass.Type)

# so luong nhan: 6 |  danh sach nhan: ['fixed acidity;"volatile acidity";"citric acid";"residual sugar";"chlorides";"free sulfur dioxide";"total sulfur dioxide";"density";"pH";"sulphates";"alcohol";"quality"']
# 1c
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
kf = KFold(n_splits=10, shuffle=True)
X = glass.iloc[:, 0:9]
y = glass.iloc[:, 9]
a = 0;
tb = 0
clf_gini = DecisionTreeClassifier(criterion="gini", random_state=100, max_depth=11, min_samples_leaf=5)
for train_index, test_index in kf.split(X):
    a = a + 1
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    clf_gini.fit(X_train, y_train)
    y_pred = clf_gini.predict(X_test)
    c = accuracy_score(y_test, y_pred)*100
    tb = tb + c
    print("Accuracy cua ",a, ":",c)

print("trung binh het",tb/10)