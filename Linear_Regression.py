"""Open-source Linear Model
Programmed by Mahir Babbar 
Fell free to use in your program """

import numpy as np
import matplotlib.pyplot as plt

#Enter your own data --------------------
np.random.seed(49)
X = np.random.randint(0, 100, size=100)
y = np.random.randint(0, 1000, size=100)
#-----------------------------------------


X_min = np.min(X)
X_max = np.max(X)
X_norm = (X - X_min) / (X_max - X_min)  


m = len(y)
alpha_values = np.array([ 0.1, 0.001, 0.0001])  #Default rate, you can adjust yours
tolerance = 1e-6
best_alpha = 0
max_iterations = 10000
best_w = 0
best_b = 0
best_cost = float('inf')


def compute_cost(X, y, w, b):
    f_wb = b + w * X
    return (1 / (2 * m)) * np.sum((f_wb - y) ** 2)


for alpha in alpha_values:
    w = 0  # set accordingly
    b = 0  # set accordingly
    previous_cost = float('inf')

    for iteration in range(max_iterations):
        f_wb = b + w * X_norm
        d_b = (1 / m) * np.sum(f_wb - y)
        d_w = (1 / m) * np.sum((f_wb - y) * X_norm)
        
        b -= alpha * d_b
        w -= alpha * d_w
        cost = compute_cost(X_norm, y, w, b)

        if cost > previous_cost or not np.isfinite(cost):
            print(f"Divergence detected at alpha={alpha}, iteration={iteration}")
            break

        if abs(previous_cost - cost) < tolerance:
            break
        previous_cost = cost

    if cost < best_cost:
        best_w, best_b, best_alpha, best_cost = w, b, alpha, cost
        


w_actual = best_w / (X_max - X_min) 
b_actual = best_b + best_w * X_min / (X_max - X_min)  

# Result Printing
print(f"LEARNING RATE : {best_alpha}")
print(f"Cost Function = {best_cost}")
print(f"Intercept (B) : {b_actual}")
print(f"Weight (w) : {w_actual}")

print(f"\nRegression Lines: f_wb={w_actual}x+{b_actual}")

# NOTE:- YOU NEED TO ADJUST THE PLOT ACCORDINGLY 
print("\n")
plt.scatter(X, y, color='red', label='Actual Data')  # Plot actual data points
plt.plot(X, b_actual + w_actual * X, color='b', label='Regression model output')  # Plot the best-fitted line on original scale
plt.xlabel("Features")
plt.ylabel("Targets")
plt.legend()
plt.show()

#BUILD YOU FUNCTION TO TEST DATA

"An Updated version for the model as an import file will be shared soon along with Regularization"
