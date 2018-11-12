# -*- coding: utf-8 -*-

# Nguyen Van Vi
# B1507343

# Cau 1
import pandas as pd
import numpy as np

glass = pd.read_csv("glass.csv")

glass.info()

so_thuoc_tinh = len(data_wine.loc[1,:])-1
so_thuoc_tinh
# Ket qua: 11
# Nhan la cot 12: quality
danh_sach_nhan = np.unique(data_wine.quality)
danh_sach_nhan
# danh sach nhan: [3, 4, 5, 6, 7, 8]

# ===
# Cau2
from sklearn.model_selection import KFold
kf = KFold(n_splits=20, shuffle=True)

X = data_wine.ix[:, 0:11]
y = data_wine.ix[:,11] #data_wine.quality
X
y

# Xac dinh so luong phan tu co trong tap test vaf tap huan luyen
for train_index, test_index in kf.split(X):
    print("Train:", train_index, "Test:", test_index)
    X_train, X_test = X.ix[train_index], X.ix[test_index]
    y_train, y_test = y.ix[train_index], y.ix[test_index]
    print("X_test:", X_test)
    print("====================")

# Cau 3

from sklearn.naive_bayes import GaussianNB

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = GaussianNB()
model.fit(X_train, y_train)
model

thucte = y_test
dubao = model.predict(X_test)
thucte
dubao

# Cau 4
from sklearn.metrics import confusion_matrix
matrix = confusion_matrix(thucte, dubao, labels=[3, 4, 5, 6, 7, 8])
matrix

y_pred = dubao
from sklearn.metrics import accuracy_score
print "Do chinh xac tong the", accuracy_score(y_test, y_pred)*100
# do chinh xac tong the: 54.79166666666667
leni = len(matrix)
accuracy_avg = 0;
for i in range(0,leni):
    accuracy = float(matrix[i][i])/(np.sum(matrix[i]))
    accuracy_avg += accuracy
    print('Do chinh xac lop '+ str(i) + ": " + str(accuracy))
accuracy_avg /= leni
accuracy_avg
# do chinh xac tung lop
#    Do chinh xac lop 0: 0.0
#    Do chinh xac lop 1: 0.0
#    Do chinh xac lop 2: 0.6634615384615384
#    Do chinh xac lop 3: 0.4852941176470588
#    Do chinh xac lop 4: 0.5777777777777777
#    Do chinh xac lop 5: 0.0

# do chinh xac trung binh: 0.2877555723143958
