import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import numpy as np

x = [[12], [16], [20], [28], [36]]
y = [[700], [900], [1300], [1750], [1800]]

plt.figure()
plt.title('Relation between diameter and price')
plt.xlabel('diameter')
plt.ylabel('price')
plt.scatter(x, y)
plt.axis([0, 50, 0, 2500])
plt.grid(True)
# plt.show()
model = LinearRegression()
model.fit(x, y)
price = model.predict(np.array([25]).reshape(-1, 1))
print('25 cm pizza should cost: $%s'%price[0][0])


x_test=[[16],[18],[22],[32],[24]]
y_test=[[1100],[850],[1500],[1800],[1100]]
score = model.score(x_test,y_test)
print('r-squared',score)


x = [[12,2], [16,1], [20,0], [28,2], [36,0]]
y = [[700], [900], [1300], [1750], [1800]]
model=LinearRegression()
model.fit(x,y)
x_test=[[16,2],[18,0],[22,2],[32,2],[24,0]]
y_test=[[1100],[850],[1500],[1800],[1100]]

price = model.predict(x_test)
for i, price in enumerate(price):
    print('Predicted:%s, Target:%s'%(price,y_test[i]))

score = model.score(x_test,y_test)
print('r-squared',score)