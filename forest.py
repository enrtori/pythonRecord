from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train , y_test = train_test_split(X,y, test_size=0.2, random_state=42)

#Arbol somple, contiene un nodo raiz con la primera decision e intermedios por ahi la hoja es el resultado 
#la profundidad son los niveles del arbol, puede dar problemas de overfitting
arbol = DecisionTreeClassifier(max_depth=3)   
arbol.fit(X_train, y_train)
print(f"Árbol accuracy: {arbol.score(X_test, y_test):.3f}")

#random forest, se ejecutan multiples arboles y se hace un quorum, se elimina overfiting pero mas pesado
bosque = RandomForestClassifier(n_estimators=100, random_state=42)
bosque.fit(X_train, y_train)
print(f"Random Forest accuracy: {bosque.score(X_test, y_test):.3f}")