from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import csv
from io import StringIO
import pandas as pd

import numpy as np
from sklearn.preprocessing import StandardScaler

with open('D:/Documents/Downloads/titanic.txt') as csvfile:
    titanic_reader = csv.reader(csvfile, delimiter=',', quotechar='"')

    row = next(titanic_reader)
    feature_names = np.array(row)

    titanic_x, titanic_y = [], []
    for row in titanic_reader:
        # print(row)
        titanic_x.append(row)
        titanic_y.append(row[2])

    titanic_x = np.array(titanic_x)
    titanic_y = np.array(titanic_y)

print(feature_names)
print(titanic_x[0], titanic_y[0])
titanic_x = titanic_x[:, [1, 4, 10]]
feature_names = feature_names[[1, 4, 10]]
print(feature_names)

print(titanic_x[12], titanic_y[12])

data = '''a,b,c,d
1.0,2.0,3.0
10.0,20.0,,30.0'''

df = pd.read_csv(StringIO(data))
print(df)
df.dropna()
df.dropna(axis=1)
df.dropna(how='all')
df.dropna(thresh=5)
df.dropna(subset=["a"])
# print(df.isnull().sum())


imp = SimpleImputer(strategy='mean')
# print(imp.fit(df).transform(df))


ages = titanic_x[:, 1]
# print(ages)
mean_age = np.mean(titanic_x[ages != 'NA', 1].astype(float))
# print(mean_age)
titanic_x[titanic_x[:, 1] == 'NA', 1] = mean_age

enc = LabelEncoder()
label_encoder = enc.fit(titanic_x[:, 2])
# print(label_encoder.classes_)
print('Categorical classes:', label_encoder.classes_)
integer_classes = label_encoder.transform(label_encoder.classes_)
print('Integer classes', integer_classes)

t = label_encoder.transform(titanic_x[:, 2])
titanic_x[:, 2] = t

enc = LabelEncoder()
label_encoder = enc.fit(titanic_x[:, 0])
print('Categorical classes:', label_encoder.classes_)
integer_classes = label_encoder.transform(label_encoder.classes_).reshape(3, 1)
print('Integer classes', integer_classes)

enc = OneHotEncoder()
one_hot_encoder = enc.fit(integer_classes)
num_of_rows = titanic_x.shape[0]
# print(num_of_rows)
t = label_encoder.transform(titanic_x[:, 0]).reshape(num_of_rows, 1)
# print(t)
new_features = one_hot_encoder.transform(t)
# print(new_features.toarray())
titanic_x = np.concatenate([titanic_x, new_features.toarray()], axis=1)
titanic_x = np.delete(titanic_x, [0], 1)
print(titanic_x)
feature_names = ['age', 'sex', 'first class', 'second class', 'third class']
titanic_x = titanic_x.astype(float)
titanic_y = titanic_y.astype(float)


# print(titanic_x)

def zscore(x: np.ndarray, axis=None):
    x_mean = x.mean(axis=axis, keepdims=True)
    x_std = np.std(x, axis=axis, keepdims=True)
    z_score = (x - x_mean) / x_std
    print('xmean', x_mean)
    print('x_std', x_std)
    return z_score


a = np.random.randint(10, size=(2, 5))
print(a)
print(zscore(a))

def min_max (x:np.ndarray, axis=None):
    x_min = x.min(axis=axis,keepdims=True)
    x_max = x.max(axis=axis,keepdims=True)
    return (x-x_min)/(x_max-x_min)

a = np.random.randint(10, size=(2, 5))
print(a)
print(min_max(a))






