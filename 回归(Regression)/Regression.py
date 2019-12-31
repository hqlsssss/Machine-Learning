import numpy as np
import matplotlib.pyplot as plt
import time

timeStart = time.time()
np.random.seed(100)
# 100 data in (0, 100)
X = np.random.random(100) * 10
# create label Y = 5X+9 with an error follow N(0,9)
Y = 5 * X + 9 + np.random.normal(0, 3, 100)

# random give a1 and a2
a1 = np.random.random()
a0 = np.random.random()

# learning rate
alpha = 0.1


# Hypothesis:
def H(theta1, theta0):
    return theta1 * X + theta0


# derivatives：
# a1: 1/m*(H-Y)*X
# a2: 1/m*(H-Y)

# object function, set m = 3
m = 3
J1 = 1 / (2 * m) * (((H(a1, a0) - Y) ** 2).sum())
J0 = 0
num = 0
J = []
a = []
b = []
# stop until converge
while (abs(J1 - J0)) > 1e-5:
    print(num, a1, a0, J1)
    J.append(J1)
    a.append(a1)
    b.append(a0)
    a1 = a1 - alpha * 1 / m * (a1 * (X ** 2).sum() + a0 * X.sum() - (X * Y).sum())
    a0 = a0 - alpha * 1 / m * (a1 * X.sum() + a0 * len(X) - Y.sum())
    J0 = J1
    J1 = 1 / (2 * m) * (((H(a1, a0) - Y) ** 2).sum())
    num += 1
print(num, a1, a0, J1)
timeEnd = time.time()
# print(Y)

plt.scatter(X, Y)
x = np.arange(0, 10, 0.1)
y = a1 * x + a0
# plt.plot(x, y, color='red', linewidth='3')
# plt.title('Simple Linear Regression')
# # plt.xlabel('size')
# # plt.ylabel('price')
# # plt.legend(('curve', 'sample'),loc='best')
# # plt.show()
plt.xlabel('iteration')
plt.ylabel('cost')
plt.xlim((0,5))
# plt.ylim((0,2))
plt.plot(range(num), J)
plt.show()
# print('time cost: ', timeEnd-timeStart)

# 3D
# from mpl_toolkits.mplot3d import Axes3D
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot(a,b,J)
# plt.xlabel('a1')
# plt.ylabel('a0')
# # plt.zlabel('num')
# plt.show()