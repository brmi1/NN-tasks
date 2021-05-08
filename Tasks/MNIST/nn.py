from NeuralNetwork import NeuralNetwork
from numpy import load, reshape, eye

with load("mnist.npz") as file:
    images, labels = file["x_train"], file["y_train"]
images = images.astype("float32") / 255
images = reshape(images, (images.shape[0], images.shape[1] * images.shape[2]))
labels = eye(10)[labels]
images.shape += (1,)
labels.shape += (1,)

network = NeuralNetwork((784, 128, 10), 10, 0.001, seed=1332)
network.train(images, labels, rate_out=1)
