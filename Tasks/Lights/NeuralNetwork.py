from numpy import random, exp, dot, array, append, argmax, round, average

class NeuralNetwork:
    def __init__(self, nn_param, n_epoch, learn_rate, seed=0):
        n_input = nn_param[0]
        n_hidden = nn_param[1]
        n_output = nn_param[2]
        self.n_epoch = n_epoch
        self.learn_rate = learn_rate

        random.seed(seed)

        self.weights_1 = random.randn(n_hidden, n_input)
        self.weights_2 = random.randn(n_output, n_hidden)

    def sigmoid(self, x, deriv=False):
        if deriv:
            return x * (1 - x)
        return 1 / (1 + exp(-x))

    def forwardprop(self, x):
        self.hidden_layer = self.sigmoid(dot(self.weights_1, x))
        self.output_layer = self.sigmoid(dot(self.weights_2, self.hidden_layer))

        return self.output_layer

    def backwardprop(self, x, ideal):
        output_layer_error = self.output_layer - ideal
        output_layer_delta = output_layer_error * self.sigmoid(self.output_layer, deriv=True)

        hidden_layer_error = dot(self.weights_2.T, output_layer_error)
        hidden_layer_delta = hidden_layer_error * self.sigmoid(self.hidden_layer, deriv=True)

        weights_2_update = dot(output_layer_delta, self.hidden_layer.T)
        weights_1_update = dot(hidden_layer_delta, x.T)

        self.weights_2 -= self.learn_rate * weights_2_update
        self.weights_1 -= self.learn_rate * weights_1_update

    def train(self, x, ideal, rate_out=500):
        for epoch in range(self.n_epoch + 1):
            err = array([])
            n_correct = 0
            for input, labels in zip(x, ideal):
                self.forwardprop(input)
                self.backwardprop(input, labels)

                err = append(err, (self.forwardprop(input) - labels) ** 2)
                n_correct += int(argmax(self.output_layer) == argmax(labels))

            if epoch % rate_out == 0:
                print(f"Epoch: {epoch} Err {round(average(err) * 100, 2)}% Acc: {round((n_correct / x.shape[0]) * 100, 2)}%")

    def predict(self, x, y=False):
        if y:
            return f"{y} => {self.forwardprop(x)}"
        return self.forwardprop(x)
