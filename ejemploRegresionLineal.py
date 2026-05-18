import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Entrada de datos Horas de estudio X y notas Y
X = np.array([[1],[2],[2],[3],[4],[4],[5],[5],[6],[7]])
y = np.array([2.8, 4.0, 5.2, 6.0, 6.7, 7.8, 7.5, 8.8, 8.5, 9.7])

# dividir en datos para entrenar y para evaluar 20% para test y aleatoriedad de 42

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#entrenar modelo:
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#comprobar datos de aprendizaje
print(f"Pendiente (w): {modelo.coef_[0]:.3f}")  
print(f"Sesgo (b):     {modelo.intercept_:.3f}") 

#Evaluar MSE con los datos que no ha llegado a ver
y_pred = modelo.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"MSE en test: {mse:.3f}")

# Predecir para nueva entrada
nuevo = np.array([[5.5]])
print(f"Predicción para 5.5: {modelo.predict(nuevo)[0]:.2f}")