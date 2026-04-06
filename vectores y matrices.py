import numpy as np

# --- Vectores ---
usuario = np.array([0.9, 0.1, 0.7, 0.3, 0.8])
print(usuario.shape)  

# --- matrices ---
X = np.array([[1,2,3],[3,4,5]])
print(X.shape)

# -- multiplicacion --
Y =np.array([[2,4],[1,2],[3,6]])
result = X @ Y

print(Y.shape)

# --transpuesta --
print(X.T)

#--inversa--
Z = np.array([[1,5],[3,8]])
Z_inv = np.linalg.inv(Z)
print(Z_inv)
