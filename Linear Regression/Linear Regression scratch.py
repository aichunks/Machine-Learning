import numpy as np
import matplotlib.pyplot as plt

dataset = [[1, 1], [2, 3], [4, 3], [3, 2], [5, 5]]
X = [row[0] for row in dataset]
Y = [row[1] for row in dataset]
# Mean X and Y
mean_x = np.mean(X)
mean_y = np.mean(Y)
 
# total number of values
n = len(X)
 
# using the formula to calculate m and c
numer = 0
denom = 0
for i in range(n):
    numer += (X[i] - mean_x) * (Y[i] - mean_y)
    denom += (X[i] - mean_x) ** 2
m= numer / denom
c = mean_y - (m * mean_x)
 
print(m)


# plotting values 
max_x = np.max(X) + 100
min_x = np.min(X) - 100
x = np.linspace(min_x, max_x, 1000)
y = c + m * x 
 
from sklearn  import metrics
r2=metrics.r2_score(x,y)
print(r2)
# Ploting Line
plt.plot(x, y, label='Regression Line')
# Ploting Scatter Points
plt.scatter(X, Y,c='r', label='Scatter Plot')
 
plt.legend()
plt.show()