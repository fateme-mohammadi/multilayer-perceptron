import numpy as np
import matplotlib.pyplot as plt
#activation function: sigmoid
def sigmoid(x):
    return 1 / (1+ np.exp(-x))


#sigmoid derivative for backprop
def sigmoid_derive(x):
    return sigmoid(x)*(1-sigmoid(x))

# Define the hyperparameters
input_size = int(input("Enter the input size: "))
hidden_size = 4        #the number of neurons in the hidden layer(s) of a neural network
output_size = 1
learning_rate = 0.1
epochs = 100

# Define the XOR input and output data
X = np.random.randint(2, size=(4, input_size))
y = np.logical_xor.reduce(X, axis=1).reshape(-1, 1).astype(int)

# Initialize the weights and biases
np.random.seed(0)
weights_input_hidden = np.random.uniform(size=(input_size, hidden_size))
biases_hidden = np.random.uniform(size=(1, hidden_size))
weights_hidden_output = np.random.uniform(size=(hidden_size, output_size))
biases_output = np.random.uniform(size=(1, output_size))

# Training loop
for epoch in range(epochs):

    #previous_weights = weights.copy()  # Make a copy of the initial weights
    
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

    # Check the stop condition
    #if previous_weights = weights:
        #break
    
# Testing the trained model
hidden_layer_input = np.dot(X, weights_input_hidden) + biases_hidden
hidden_layer_output = sigmoid(hidden_layer_input)
output_layer_input = np.dot(hidden_layer_output, weights_hidden_output) + biases_output
output_layer_output = sigmoid(output_layer_input)

print("Input Data:")
print(X)
print("Truth Output:")
print(y)
print("Predicted Output:")
print(output_layer_output)


