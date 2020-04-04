from sklearn import datasets
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

print(iris.data, iris.target)
# print(iris.data.shape)
# print(len(iris.data))

clf = svm.SVC(gamma='auto')

clf.fit(iris.data, iris.target)

print(clf.predict([iris.data[1]]))

x, y = iris.data, iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=1)

model = svm.SVC()

model.fit(x_train, y_train)

pred = model.predict(x_test)
print(accuracy_score(y_test, pred))

print(model.predict([[1.4, 3, 5, 0.2]]))
