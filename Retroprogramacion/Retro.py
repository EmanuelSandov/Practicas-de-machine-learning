import numpy as np

# Función de activación sigmoide
def sigmoid(x):
    return 1/(1+np.exp(-x))

# Derivada de la función de activación sigmoide
def sigmoid_derivada(x):
    return x*(1-x)

# Datos de entrada (X) y salida esperada (y)
X = np.array([[0,0],  # Entradas
              [0,1],
              [1,0],
              [1,1]])

y = np.array([[0],  # Salidas deseadas
              [1],
              [1],
              [0]])

# Inicialización de pesos y sesgos (bias)
np.random.seed(1)
input_neurons = 2
hidden_neurons = 3
output_neurons = 1

# Pesos entre la capa de entrada y la capa oculta
weights_input_hidden = np.random.uniform(size=(input_neurons, hidden_neurons))
# Pesos entre la capa oculta y la capa de salida
weights_output_hidden = np.random.uniform(size=(hidden_neurons, output_neurons))

# Sesgos para la capa oculta y la capa de salida
bias_hidden = np.random.uniform(size=(1, hidden_neurons))
bias_output = np.random.uniform(size=(1, output_neurons))

# Hiperparámetros
epochs = 10000
learning_rate = 0.1

# Entrenamiento del modelo
for epoch in range(epochs):
    # Feedforward
    # Calcular la entrada de la capa oculta
    input_hidden = np.dot(X, weights_input_hidden) + bias_hidden
    # Aplicar la función de activación sigmoide
    output_hidden = sigmoid(input_hidden)

    # Calcular la entrada de la capa de salida
    input_output = np.dot(output_hidden, weights_output_hidden) + bias_output
    # Aplicar la función de activación sigmoide
    output = sigmoid(input_output)

    # Retropropagación
    # Calcular el error de la salida
    error = y - output

    # Calcular los deltas y ajustar los pesos
    delta_output = error * sigmoid_derivada(output)
    error_hidden = delta_output.dot(weights_output_hidden.T)
    delta_hidden = error_hidden * sigmoid_derivada(output_hidden)

    # Actualizar los pesos y los sesgos
    weights_output_hidden += output_hidden.T.dot(delta_output) * learning_rate
    bias_output += np.sum(delta_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(delta_hidden) * learning_rate
    bias_hidden += np.sum(delta_hidden, axis=0, keepdims=True) * learning_rate

# Resultado final después del entrenamiento
print("Resultado después del entrenamiento")
print(output)