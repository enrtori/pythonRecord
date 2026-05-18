import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

#ejemplos media y desviacion
muestras = np.random.normal(loc=1, scale=5, size=100)
print(f"Media: {muestras.mean():.3f}")
print(f"Desviacion: {muestras.std():.3f}")

#ejemplo de pesos de red neuronal
pesos = np.random.normal(loc=0, scale=0.01, size=(128, 64))
print(f"Shape pesos: {pesos.shape}")       