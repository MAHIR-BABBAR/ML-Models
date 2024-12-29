"""Open-source Linear Model
Programmed by Mahir Babbar 
Fell free to use in your program """

import numpy as np
import matplotlib.pyplot as plt



# Enter Your Data ------------------------------------------------------------------------------
np.random.seed(50)
age = np.random.randint(20, 71, 100)
experience = np.random.randint(1, 51, 100)

X = np.column_stack((age, experience))
y = np.array([1 if (a > 35 and e > 20) else 0 for a, e in zip(age, experience)])
#-------------------------------------------------------------------------


X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X_scaled = (X - X_min) / (X_max - X_min)




alpha_values = np.array([0.1, 0.01, 0.001])  #Default Learning rates, you can adjust to yours
best_w = np.zeros(X.shape[1])  
best_b = 0
best_cost = float('inf')
max_iterations = 1000
tolerance = 1e-6


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_cost(y_predicted, y):
    m = y.shape[0]
    cost = - (1/m) * np.sum(y * np.log(y_predicted) + (1 - y) * np.log(1 - y_predicted))
    return cost

def compute_gradient_logistic(X, y, w, b):
    m, n = X.shape
    dj_dw = np.zeros((n,))    
    dj_db = 0.  

    for i in range(m):
        f_wb_i = sigmoid(np.dot(X[i], w) + b)
        err_i = f_wb_i - y[i]
        for j in range(n):
            dj_dw[j] += err_i * X[i, j]
        dj_db += err_i
    dj_dw /= m
    dj_db /= m

    return dj_db, dj_dw


for alpha in alpha_values:
    w = np.zeros(X.shape[1])  #default parameters
    b = 0   #default parameters
    previous_cost = float('inf')
    for i in range(max_iterations):
        d_b, d_w = compute_gradient_logistic(X_scaled, y, w, b)
        w -= alpha * d_w
        b -= alpha * d_b
        y_pred = sigmoid(np.dot(X_scaled, w) + b)
        cost = compute_cost(y_pred, y)
        if abs(previous_cost - cost) < tolerance:
            break
        previous_cost = cost
    if cost < best_cost:
        best_w, best_b, best_alpha, best_cost = w, b, alpha, cost



w_original = w / (X_max - X_min)  
b_original = b - np.dot(X_min, w_original) 



print(f"LEARNING RATE: {best_alpha}")
print(f"Cost Function = {best_cost}")
print(f"Intercept (B) : {b_original}")
print(f"Weights (w) : {w_original}")

print(f"Final f(w, b) equation: P(Purchase=1) = sigmoid({w_original[0]:.4f}*Age + {w_original[1]:.4f}*Income + {b_original:.4f})")



# Note:- Adjust your plot accordingly

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='k', label='Original Data')


plt.scatter(X[y == 1, 0], X[y == 1, 1], color='blue', marker='o', label='Purchased = 1')
plt.scatter(X[y == 0, 0], X[y == 0, 1], color='red', marker='x', label='Purchased = 0')  

plt.xlabel('Age')
plt.ylabel('Income')
plt.title(f'Logistic Regression (2D) with Learning Rate {best_alpha}')

xx, yy = np.meshgrid(np.linspace(X_min[0], X_max[0], 100), np.linspace(X_min[1], X_max[1], 100))
grid_points = np.c_[xx.ravel(), yy.ravel()]
grid_points_scaled = (grid_points - X_min) / (X_max - X_min)
z = sigmoid(np.dot(grid_points_scaled, best_w) + best_b)
z = z.reshape(xx.shape)


plt.contourf(xx, yy, z, levels=np.linspace(0, 1, 100), cmap=plt.cm.coolwarm_r, alpha=0.3)

plt.colorbar(label='Probability of Purchasing')
plt.legend()
plt.show()

#BUILD YOU FUNCTION TO TEST DATA

"An Updated version for the model as an import file will be shared soon along with Regularization"