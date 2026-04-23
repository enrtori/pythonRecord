import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

#Datos no claros
np.random.seed(0)
X = np.sort(np.random.rand(30, 1) * 6, axis=0)
y = np.sin(X).ravel() + np.random.normal(0, 0.3, X.shape[0])

#dividir
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#Division y prueba de distintos modelos, para diferentes grados
#convierte en variables polinomicas y se aplica regresion sobre ello
for grado in [1, 4, 15]:
    pipe = Pipeline([
        ('poly', PolynomialFeatures(degree=grado)),
        ('model', LinearRegression())
    ])
    pipe.fit(X_train, y_train)
    train_err = mean_squared_error(y_train, pipe.predict(X_train))
    test_err  = mean_squared_error(y_test,  pipe.predict(X_test))
    print(f"Grado {grado:2d} → train MSE: {train_err:.3f}  |  test MSE: {test_err:.3f}")
#Los resultados son: g1 tr:0.280 te:0.162 el modelo es una linea y no aprende bien con los datos, underfitting
#g4 tr:0.127 te:0.027 error bajo en entrenamiento y mas bajo en test, modelo perfecto
#g15 tr:0.089 te:0.048 error muy bajo en entrenamiento pero mas alto que el anterior en test. Overfitting, ha haprendido demasiado