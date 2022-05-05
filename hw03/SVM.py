from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier

from sklearn import metrics
import numpy as np



data_np = np.loadtxt("theta_090.csv",delimiter=",")

# print(data_np.shape)
label = []
for i in range(5):
    label = label + [i]*40

X_train, X_test, y_train, y_test = train_test_split(data_np, label, test_size=.2, random_state=10)
# 训练模型
model = OneVsRestClassifier(svm.SVC(kernel='linear', probability=True))

clt = model.fit(X_train, y_train)


y_test_pred = clt.predict(X_test)
ov_acc = metrics.accuracy_score(y_test_pred, y_test)
print(f"overall accuracy:{ov_acc:.4f}")

acc_for_each_class = metrics.precision_score(y_test, y_test_pred, average=None)
print(f"acc_for_each_class:\n")
print("{} {} {} {} {} {}".format(*acc_for_each_class,ov_acc))


