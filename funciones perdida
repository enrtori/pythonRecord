import numpy as np

# ── Clasificación BINARIA (sigmoid en salida) 
# Binary Cross-Entropy
def bce_loss(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  
    return -np.mean(y_real * np.log(y_pred) + (1 - y_real) * np.log(1 - y_pred))

y_real = np.array([1, 0, 1, 1])
y_pred = np.array([0.9, 0.1, 0.8, 0.6])
print(f"BCE Loss: {bce_loss(y_real, y_pred):.4f}")   

# ── Clasificación MULTICLASE (softmax en salida)
# Categorical Cross-Entropy
def cce_loss(y_real, y_pred):
    y_pred = np.clip(y_pred, 1e-7, 1)
    return -np.sum(y_real * np.log(y_pred))

y_real = np.array([0, 0, 1, 0])          
y_pred = np.array([0.05, 0.1, 0.8, 0.05])
print(f"CCE Loss: {cce_loss(y_real, y_pred):.4f}")

# ── Regresión (sin activación o ReLU en salida) 
# Mean Squared Error
def mse_loss(y_real, y_pred):
    return np.mean((y_real - y_pred) ** 2)

y_real = np.array([3.5, 7.0, 2.1])
y_pred = np.array([3.2, 7.4, 1.9])
print(f"MSE Loss: {mse_loss(y_real, y_pred):.4f}")   