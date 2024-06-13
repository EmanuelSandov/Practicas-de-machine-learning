import numpy as np
import matplotlib.pyplot as plt

# Datos de 10 personas representados como una matriz de características [edad, ahorro]
personas = np.array([[0.3, 0.4], [0.4, 0.3],
                     [0.3, 0.2], [0.4, 0.1],
                     [0.5, 0.2], [0.4, 0.8],
                     [0.6, 0.8], [0.5, 0.6],
                     [0.7, 0.6], [0.8, 0.5]])

# Clases que indican si una tarjeta es aprobada (1) o denegada (0) para cada persona
clases = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

# Crear la gráfica de dispersión para visualizar los datos
for i, clase in enumerate(clases):
    if clase == 1:
        plt.scatter(personas[i, 0], personas[i, 1], color='green', marker='o',
                    label='Aprobada' if i == clases.tolist().index(1) else "")
    else:
        plt.scatter(personas[i, 0], personas[i, 1], color='brown', marker='x',
                    label='Denegada' if i == clases.tolist().index(0) else "")

# Etiquetas y título del gráfico
plt.xlabel('Edad')
plt.ylabel('Ahorro')
plt.title('¿Tarjeta Platinum?')
plt.legend()
plt.grid(True)

# Mostrar la gráfica
plt.show()

# Función de activación para el perceptrón
def activacion(peso, x, b):
    z = peso * x
    if z.sum() + b > 0:
        return 1
    else:
        return 0

# Inicialización de pesos y bias aleatorios
pesos = np.random.uniform(-1, 1, size=2)
b = np.random.uniform(-1, 1)

# Prueba de la función de activación con dos puntos
print(f'Funcion de activacion [0.1,0.7]: {pesos, b, activacion(pesos, [0.1, 0.7], b)}')
print(f'Funcion de activacion [0.6,0.8]: {pesos, b, activacion(pesos, [0.6, 0.8], b)}')

# Entrenamiento del perceptrón
pesos = np.random.uniform(-1, 1, size=2)
b = np.random.uniform(-1, 1)
tasa_de_aprendizaje = 0.01
epocas = 100

#print("\nEntrenamiento del perceptron del [0.5, 0.5]")
#print("\nEntrenamiento del perceptron del [0.1, 0.7]")
print("\nEntrenamiento del perceptron del [0.6, 0.8]")

# Iterar a través de las épocas de entrenamiento
for epoca in range(epocas):
    error_total = 0
    for i in range(len(personas)):
        # Predicción utilizando la función de activación
        prediccion = activacion(pesos, personas[i], b)
        # Calcular el error
        error = clases[i] - prediccion
        # Sumar el error cuadrático total
        error_total += error ** 2
        # Actualización de los pesos y el bias
        pesos[0] += tasa_de_aprendizaje * personas[i][0] * error
        pesos[1] += tasa_de_aprendizaje * personas[i][1] * error
        b += tasa_de_aprendizaje * error
    # Imprimir el error total por época
    print(error_total, end=" ")

# Realizar una predicción final después del entrenamiento
#print(f'\nActivacion para [0.5, 0.5]: {activacion(pesos, [0.5, 0.5], b)}')
#print(f'\nActivacion para [0.1, 0.7]: {activacion(pesos, [0.1, 0.7], b)}')
print(f'\nActivacion para [0.6, 0.8]: {activacion(pesos, [0.6, 0.8], b)}')