from NeuralNetwork import NeuralNetwork
from numpy import array

data = array([[0, 0, 1, 0],
              [0, 1, 1, 1],
              [0, 1, 0, 1],
              [0, 1, 1, 1],
              [1, 0, 0, 0]])

x = data[:,:3]
ideal = data[:,-1:]
x.shape += (1,)
ideal.shape += (1,)

network = NeuralNetwork((3, 5, 1), 500, 0.1)
network.train(x, ideal, rate_out=50)
