# Anubhav Parbhakar

from math import prod
import random
import math


class Layer:
    def __init__(self, is_first_layer: bool, is_output_layer: bool):
        self.ifl = is_first_layer
        self.iol = is_output_layer

    def is_first_layer(self) -> bool:
        return self.ifl

    def is_output_layer(self) -> bool:
        return self.iol

    def create_layer(self):
        # each row i represents the # of neurons in the curr. layer
        matrix = []
        for i in range(9):
            temp = []
            if self.ifl:
                for j in range(10):
                    temp.append(random.random())
                matrix.append(temp)
            else:
                for j in range(9):
                    temp.append(random.random())
                matrix.append(temp)
        self.matrix = matrix

    def create_bias(self):
        bias = []
        for i in range(9):
            temp = []
            for j in range(1):
                temp.append(random.random())
            bias.append(temp)
        self.bias = bias

    def get_bias(self) -> list:
        return self.bias

    def multiply_weight(self, input: list) -> list:
        product = []
        matrix = self.matrix
        self.input = input
        for i in range(len(matrix)):
            temp = []
            total = 0
            for j in range(len(matrix[i])):
                total += matrix[i][j] * input[j][0]
            temp.append(total)
            product.append(temp)
        self.product = product
        return product

    def add_bias(self, product: list) -> list:
        up = []
        bias = self.bias
        for i in range(len(product)):
            temp = []
            total = 0
            for j in range(1):
                total += product[i][j] + bias[i][j]
            temp.append(total)
            up.append(temp)
        self.up = up
        return up

    def sigmoid_activate(self, product) -> list:
        output = []
        for i in range(len(product)):
            temp = []
            for j in range(1):
                x = 1/(1 + (math.e ** (-1 * product[i][j])))
                temp.append(x)
            output.append(temp)
        self.output = output
        return output

    def relu_activate(self, product) -> list:
        output = []
        for i in range(len(product)):
            temp = []
            for j in range(1):
                temp.append(max(0, product[i][j]))
            output.append(temp)
        self.output = output
        return output

    def set_error(self, error: list):
        self.error = error

    def set_error_delta(self, error_delta: list):
        self.error_delta = error_delta

    def sigmoid_derivative(self, x: int) -> int:
        y = x * (1 - x)
        return y

    def relu_derivative(self, x: int) -> int:
        if x < 0:
            return 0
        return 1

    def set_adj_amount(self):
        adj_amount = []
        for i in range(len(self.error_delta)):
            temp = []
            for j in range(1):
                for k in range(len(self.input)):
                    for l in range(1):
                        x = self.error_delta[i][j] * self.input[k][l]
                        temp.append(x)
            adj_amount.append(temp)
        self.adj_amount = adj_amount

    def adjust_weights(self):
        adj_matrix = []
        for i in range(len(self.matrix)):
            temp = []
            for j in range(len(self.matrix)):
                x = self.matrix[i][j] - self.adj_amount[i][j]
                temp.append(x)
            adj_matrix.append(temp)
        self.matrix = adj_matrix

    def adjust_bias(self):
        adj_bias = []
        for i in range(len(self.bias)):
            temp = []
            for j in range(1):
                x = self.bias[i][j] - self.error_delta[i][j]
                temp.append(x)
            adj_bias.append(temp)
        self.bias = adj_bias
