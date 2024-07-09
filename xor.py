import numpy as np
import matplotlib.pyplot as plt
#activation function: sigmoid
def sigmoid(x):
    return 1 / (1+ np.exp(-x))


#sigmoid derivative for backprop
def sigmoid_derive(x):
    return sigmoid(x)*(1-sigmoid(x))

#forward function
def forward(x, w1, w2, predict = False):
    a1 = np.matmul(x,w1)
    z1 = sigmoid(a1)

    #creat and add bias
    bias = np.ones((len(z1),1))
    z1 = np.concatenate((bias,z1), axis = 1)
    a2 = np.matmul(z1,w2)
    z2 = sigmoid(a2)
    if predict:
        return z2
    return a1, z1, a2, z2

#backprop function
def backprop(a2, z0, z1, z2, y):
    delta2 = z2 - y
    Delta2 = np.matmul.dot(z1.T,delta2)
    delta1 = (delta2.dot(w2[1:,:].T))*sigmoid_derive(a1)
    Delta1 = np.matmul(z0.T,delta1)
    return delta2, Delta2, Delta1


#first column = bias
x = np.array([[1,1,0],
              [1,0,1],
              [1,0,0],
              [1,1,1]])
#output
y = np.array([[1],[1],[0],[0]])

#init weights
w1 = np.randm.randn(3,5)
w2 = np.randm.randn(6,1)

#init learning rate
lr = 0.09

costs = []

#init epochs
epochs = 15000

m = len(x)

#start training
for i in range(epochs):

    #forward
    a1, z1, a2, z2 = forward(x, w1, w2)

    #backprop
    delta2, Delta2, Delta1 = backprop(a2, x, z1, z2, y)

    w1 -= lr*(1/m)*Delta1
    w2 -= lr*(1/m)*Delta2

    #add costs to list for plotting
    c = np.mean(np.abs(delta2))
    costs.append(c)

    if i % 1000 == 0:
        print(f"Iteration: {i}. Error: {c}")

#training comlate
print("Traning complete.")

#make predictions
z3 = forward(x, w1, w2, True)
print("Percentages: ")
print(z3)
print("Percentages: ")
print(np.round(z3))

#plot cost
plt.plot(costs)
plt.show()
