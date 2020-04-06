import numpy as np
import matplotlib.pyplot as plt

x = np.array([-1.2, -0.6, 0, 0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2])
y = np.cos(x)
z = np.polyfit(x, y, 3)
p = np.poly1d(z)
z30 = np.polyfit(x, y, 30)
p30 = np.poly1d(z30)
# plt.plot(x, y)
# plt.show()
print(z, p)
xp = np.linspace(-2, 5, 100)
plt.plot(xp, np.cos(xp), '.', xp, p(xp), '-', xp, p30(xp), '*')
plt.ylim(-3,3)
plt.grid(True)

plt.show()
