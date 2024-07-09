import numpy as np

# Define the sigmoid activation function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Define the derivative of the sigmoid function
def sigmoid_derivative(x):
    return x * (1 - x)

#FORWARD FUNCTION
def forward (x, w1, w2, predict=False):
    a1 = np.matmul(x, w1)
    z1 = sigmoid(a1)
    
    #create and add bias
    bias = np.ones(len(z1),1)
    z1 = np.concatenate((bias, z1), axis=1)
    a2 = np.matmul(z1, w2)
    z2 = sigmoid(a2)
    
    if predict:
        return z2
    
    return a1, z1, a2, z2
    
def backprop(a2, z0, z1, z2, y):
    delta2 = z2-y
    Delta2 = np.matmul(z1.T, delta2)
    delta1 = (delta2.dot(w2[1:,:].T))*sigmoid_derivative(a1)
    Delta1 = np.matmul(z0.T, delta1)

    return delta2, Delta1, Delta2

# Define the XOR input and output data
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define the hyperparameters
input_size = 2
hidden_size = 4
output_size = 1
learning_rate = 0.1
epochs = 10000

# Initialize the weights and biases
np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
biases_hidden = np.random.uniform(size=(1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
biases_output = np.random.uniform(size=(1, output_size))

# Training loop
for epoch in range(epochs):
    # Forward propagation
    hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
    output_layer_output = sigmoid(output_layer_input)

    # Backpropagation
    error = y - output_layer_output
    d_output = error * sigmoid_derivative(output_layer_output)
    error_hidden = d_output.dot(weights_hidden_output.T)
    d_hidden = error_hidden * sigmoid_derivative(hidden_layer_output)

    # Update weights and biases
    weights_hidden_output += hidden_layer_output.T.dot(d_output) * learning_rate
    biases_output += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    weights_input_hidden += X.T.dot(d_hidden) * learning_rate
    biases_hidden += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

# Testing the trained model
hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
output_layer_output = sigmoid(output_layer_input)

print("Predicted Output:")
print(output_layer_output)
