from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# dataset de diagnortico de cancer real, lo primero dividir los valores entre test y entrenamiento
X,y = load_breast_cancer(return_X_y = True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Normalizar:
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test) 

modelo = LogisticRegression()
modelo.fit(X_train,y_train)

y_pred = modelo.predict(X_test)
print(classification_report(y_test, y_pred))