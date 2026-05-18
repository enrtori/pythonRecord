import numpy as np

notas = np.array([6.0, 7.5, 8.0, 5.0, 9.0, 7.0, 6.5, 8.5])

media = np.mean(notas)
varianza = np.var(notas)
desviacion = np.std(notas)

print(f'media = {media:.2f}')
print(f'varianza = {varianza:.2f}')
print(f'desviacion tipica = {desviacion:.2f}')