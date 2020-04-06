from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from math import sqrt
from sklearn.metrics import mean_squared_error

boston = datasets.load_boston()

print(dir(boston))

x = boston.data
y = boston.target

# print(x)
# print(y)

df_x = pd.DataFrame(x)
df_y = pd.DataFrame(y)

df_x.columns = boston.feature_names
df_y.columns = ["target"]
df = pd.concat([df_x, df_y], axis=1)
corr = df.corr()
print(corr)

lstat = df_x.loc[:, "LSTAT"]
print(type(lstat))
print(lstat.shape)
lstat = lstat.values
lstat = lstat.reshape(-1, 1)
print(lstat)
plt.scatter(lstat, y)
# plt.show()

x_train, x_test, y_train, y_test = train_test_split(lstat, y, test_size=0.3, random_state=0)

lr = LinearRegression()
lr.fit(x_train, y_train)

plt.scatter(lstat, y)
plt.plot(x_test, lr.predict(x_test), color='red')
plt.title('boston housing')
plt.xlabel('lstat')
plt.ylabel('target')

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)
print('RMSE test', sqrt(mean_squared_error(y_test, y_test_pred)))
print('R^2 Train : %.3f, Test : %.3f' % (lr.score(x_train, y_train), lr.score(x_test, y_test)))

plt.show()
