import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Datos de entrenamiento
fahrenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)
celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)

# Definición de la capa y el modelo
capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

# Compilación del modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

# Entrenamiento del modelo
print("Comienza entrenamiento...")
historial = modelo.fit(fahrenheit, celsius, epochs=1000, verbose=False)
print("Modelo entrenado")

# Visualización de la pérdida durante el entrenamiento
plt.xlabel("# Epoca")
plt.ylabel("Magnitud de pérdida")
plt.plot(historial.history['loss'])
plt.show()

# Predicción
print("Predicción")
resultado = modelo.predict(np.array([100.0]))
print("El resultado es: " + str(resultado[0][0]) + " Celsius!")

#Variables internas
print("Variables internas")
print(capa.get_weights())