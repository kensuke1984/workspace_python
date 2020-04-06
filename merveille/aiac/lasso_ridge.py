from sklearn.linear_model import Lasso, Ridge, LinearRegression
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

boston = load_boston()

x = boston.data
y = boston.target

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=123)
model0 = LinearRegression()
model0.fit(x_train, y_train)
pred0 = model0.predict(x_test)
linear_score = r2_score(y_test, pred0)
model1 = Lasso()
model1.fit(x_train, y_train)
pred1 = model1.predict(x_test)

lasso_score = r2_score(y_test, pred1)

model2 = Ridge()
model2.fit(x_train, y_train)
pred2 = model2.predict(x_test)
ridge_score = r2_score(y_test, pred2)

print("Linear regression", linear_score)
print("lasso", lasso_score)
print("ridge", ridge_score)

plt.plot(model0.predict(x_test), linestyle='solid', color='red', label='lr')
plt.plot(model1.predict(x_test), linestyle='solid', color='green', label='lasso')
plt.plot(model2.predict(x_test), linestyle='solid', color='blue', label='ridge')
plt.title('lin, lasso, ridge')
plt.legend()
plt.show()
