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

z1 = w1 * x
a = sigmoid(z1)
z2 = w2 * a
L = (z2 - y) **2

print(L)

# Backprop manual (regla de la cadena)
dL_dz2 = 2 * (z2 - y)
dz2_da = w2
da_dz1 = a * (1 - a)          
dz1_dw1 = x

dL_dw1 = dL_dz2 * dz2_da * da_dz1 * dz1_dw1   # regla de la cadena
print(f"Gradiente de w1: {dL_dw1:.4f}")