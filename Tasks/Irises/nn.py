from NeuralNetwork import NeuralNetwork
from numpy import loadtxt

data = loadtxt('iris.csv', delimiter=',')
x = data[:,:4]
ideal = data[:,-3:]
x.shape += (1,)
ideal.shape += (1,)

network = NeuralNetwork((4, 8, 3), 500, 0.01, seed=1332)
network.train(x, ideal, rate_out=100)
