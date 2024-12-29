"""Open-source Linear Model
Programmed by Mahir Babbar 
Fell free to use in your program """

import numpy as np
import matplotlib.pyplot as plt

# Enter you data -----------------------------------------------------------------
np.random.seed(1)
X1 = np.random.randint(0, 10000, size=100)  
X2 = np.random.randint(0, 10000, size=100)  
X3 = np.random.randint(0, 10000, size=100)  

y = 2 * X1 + 3 * X2 + 4 * X3 + np.random.randint(500, 1000, size=100)  

X = np.column_stack((X1, X2, X3)) 
#--------------------------------------------------------------------------------



X_min = np.min(X, axis=0)
X_max = np.max(X, axis=0)
X_norm = (X - X_min) / (X_max - X_min) 




m = len(y)
alpha_values = np.array([1, 0.1, 0.001, 0.0001])  #Default Learning rates, you can adjust to yours
tolerance = 1e-6
best_alpha = 0
max_iterations = 10000
best_w = np.zeros(X.shape[1]) 
best_b = 0
best_cost = float('inf')




def compute_cost(X, y, w, b):
    f_wb = b + np.dot(X, w)  
    return (1 / (2 * m)) * np.sum((f_wb - y) ** 2)


def compute_gradient(X, y, w, b): 
   
    m,n = X.shape          
    dj_dw = np.zeros((n,))
    dj_db = 0.

    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
        
    return dj_db, dj_dw




for alpha in alpha_values:
    w = np.zeros(X.shape[1])  #Default initial rates
    b = 100  #Default initial rates
    previous_cost = float('inf')

    for iteration in range(max_iterations):
        
        d_b,d_w=compute_gradient(X_norm,y,w,b)
       
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
b_actual = best_b - np.dot(X_min, w_actual)  





# Printing Data, format according to yours
print(f"LEARNING RATE : {best_alpha}")
print(f"Cost Function = {best_cost}")
print(f"Intercept (B) : {b_actual}")
print(f"Weights (w) : {w_actual}")

if len(w) == 1: 
    print(f"Final Regression Line: y = {b:.4f} + {w_actual[0]:.4f} * X")
else:  
    equation = "y = " + f"{b_actual:.4f}"
    for i in range(len(w)):
        equation += f" + {w_actual[i]:.4f} * X{i+1}"  
    print(f"Final Regression Line: {equation}")



#Note: Plotting muliple regrssion requires advanced setting, Set the plot accordingly

#The plot used here is 2 feature plot with target

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X1, X2, y, color='red', label='Actual Data')

y_pred = b_actual + np.dot(X, w_actual)  
ax.plot_trisurf(X1, X2, y_pred, color='blue', alpha=0.5, label='Predicted Model')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.legend()
plt.show()

#BUILD YOU FUNCTION TO TEST DATA

"An Updated version for the model as an import file will be shared soon along with Regularization"