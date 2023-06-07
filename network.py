# Anubhav Parbhakar

from layer import Layer


class Network:
    def __init__(self):
        self.layers = []

    def add_layer(self, is_first_layer: bool, is_output_layer: bool):
        layer = Layer(is_first_layer, is_output_layer)
        layer.create_layer()
        layer.create_bias()
        self.layers.append(layer)

    def print_layers(self) -> list:
        return self.layers

    def set_error_outer_layer(self, layer: Layer, exp_output: list):
        error = []
        for i in range(len(layer.output)):
            temp = []
            for j in range(1):
                x = 2 * (layer.output[i][j] - exp_output[i][j])
                temp.append(x)
            error.append(temp)
        layer.set_error(error)

    def set_error_hidden_layer(self, current: Layer, prev: Layer):
        error = []
        for i in range(len(prev.matrix)):
            temp = []
            total = 0
            for j in range(len(prev.error_delta[i])):
                total += prev.matrix[i][j] * prev.error_delta[j][0]
            temp.append(total)
            error.append(temp)
        current.set_error(error)

        return error

    def set_error_delta(self, layer: Layer):
        error_delta = []
        for i in range(len(layer.error)):
            temp = []
            for j in range(1):
                # can use relu_derivative here instead as well
                # note: if sigmoid was used for classify, sigmoid' should be used
                # for back progagation, same for relu and relu'
                x = layer.error[i][j] * \
                    layer.sigmoid_derivative(layer.output[i][j])
                temp.append(x)
            error_delta.append(temp)
        layer.set_error_delta(error_delta)
