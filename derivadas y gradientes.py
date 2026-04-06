import numpy as np

# --derivada numerica --
def L(w):
    return w**2
def grad_L(w):
    return 2*w

#gradient desc
w= 3.0
lr = 0.3

for i in range(10):
    g = grad_L(w)
    w = w -lr *g
    print(f"Paso {i+1}: w={w:.4f}, L={L(w):.4f}")

# --- Regla de la cadena 
# L = (w2 * sigmoid(w1 * x) - y)^2
def sigmoid(X):
    return 1/(1 + np.exp(X))

x, y = 2.0, 1.0
w1, w2 = 0.5, 0.8

z = w1 * x
z = sigmoid(z)
z = w2 * z
z = (z - y) **2

print(z)
