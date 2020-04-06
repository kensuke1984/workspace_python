from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

dataset = load_iris()
x, y = dataset.data, dataset.target

x_train, x_test, y_train, y_test = train_test_split(x, y)

model=LogisticRegression(solver='liblinear')
model.fit(x_train,y_train)
pred = model.predict(x_test)
acc=accuracy_score(y_test,pred)
print(acc)
