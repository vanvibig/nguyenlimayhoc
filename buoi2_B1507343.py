# Nguyen Van Vi
# B1507343

# Cau 1
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
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
        glass.iloc[:, 0:9], 
        glass.Type, 
        test_size=1/3.0, 
        random_state=5)
len(X_test) 
# so phan tu test: 178
label_uniqe = np.unique(y_test)
label_uniqe
# so nhan cua tap test: array([3, 4, 5, 6, 7], dtype=int64)
# 1d
from sklearn.tree import DecisionTreeClassifier
clf_gini = DecisionTreeClassifier(
        criterion="gini", 
        random_state=40, 
        max_depth=3, 
        min_samples_leaf=5)
clf_gini.fit(X_train, y_train)
# 1e
y_pred = clf_gini.predict(X_test)
from sklearn.metrics import accuracy_score
print( "Accuracy is", accuracy_score(y_test, y_pred)*100)
# do chinh xac tong the: 61.79775280898876
from sklearn.metrics import confusion_matrix
#confusion_matrix(y_test, y_pred, labels=label_uniqe)
#       [[ 0,  0,  1,  0,  0],
#        [ 0,  0,  1,  2,  0],
#        [ 0,  0, 70, 18,  0],
#        [ 0,  0, 25, 33,  5],
#        [ 0,  0,  1, 15,  7]]

reality = y_test
predict = y_pred
labels = np.unique(glass.Type)
matrix = confusion_matrix(reality, predict, labels=label_uniqe)
matrix

leni = len(matrix)
accuracy_avg = 0;
for i in range(0,leni):
    accuracy = float(matrix[i][i])/(np.sum(matrix[i]))*100
    accuracy_avg += accuracy
    print('Do chinh xac lop '+ str(i) + ": " + str(accuracy) + "%")
print("Do chinh xac tong the:",accuracy_avg/leni, "%")


#========================
# predict input



































