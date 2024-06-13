import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

#Generar datos sinteticos
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

#Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Crear un pipeline con transformacion polimonial y regresion logica
model = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('logistic_regression', LogisticRegression())
])

#Entrenar el modelo
model.fit(X_train, y_train)

#Aprender y evaluar el modelo
y_pred = model.predict(X_test)
print("Accuracy: ", accuracy_score(y_test, y_pred))


#Funcion para graficar la frontera de decision
def graficar(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap="cool")
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, edgecolors='k', cmap="cool")
    plt.title("Frontera de decision")
    plt.xlabel("Caracteristica 1")
    plt.ylabel("Caracteristica 2")
    plt.show()


#Graficar la frontera de decision
graficar(model, X, y)
