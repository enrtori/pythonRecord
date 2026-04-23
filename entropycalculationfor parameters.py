import numpy as np

# probabilidades para 3 clases: [gato, perro, pájaro]
y_real      = np.array([1, 0, 0])      # respuesta correcta es "gato"
y_predicho  = np.array([0.7, 0.2, 0.1])  # el modelo dice 70% gato

cross_entropy = -np.sum(y_real*np.log(y_predicho))
print(f'Acertado del modelo: {cross_entropy:.2f}. Cuanto mas bajo mejor.')

y_predicho2 = np.array([0.9,0.1,0.1])
cross_entropy2 = -np.sum(y_real*np.log(y_predicho2))
print(f'Acertado del modelo: {cross_entropy2:.2f}. Si sube {y_predicho2[0]} y no {y_predicho2[1]} y {y_predicho2[2]}. La prediccion mejora.')

y_predicho3 = np.array([0.9,0.8,0.9])
cross_entropy3 = -np.sum(y_real*np.log(y_predicho3))
print(f'Acertado del modelo: {cross_entropy3:.2f}. Si se mantiene {y_predicho3[0]} aunque suban  la certeza de  {y_predicho3[1]} y {y_predicho3[2]}. La prediccion no varia porque solo se tiene en cuenta [1,0,0].')
