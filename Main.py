# Anubhav Parbhakar

import random
from network import Network
from layer import Layer
import random


def state_to_board(state: list) -> list:
    actual_board = state[1:]
    board = []
    temp = []
    counter = 0

    for i in range(len(actual_board) + 1):
        if counter == 3:
            board.append(temp.copy())
            temp.clear()
            counter = 0
        if(i == len(actual_board)):
            return board
        temp.insert(counter, actual_board[i])
        counter += 1


def board_to_state(board: list, turn: int) -> list:
    state = []
    state.append(turn)
    for i in board:
        state.extend(i)
    return state


def to_move(state: list) -> str:
    if state[0] == 0:
        return "white"
    else:
        return "black"


def is_terminal(state: list) -> bool:
    board = state_to_board(state)
    for i in board[0]:
        if i == 1:
            return True

    for i in board[2]:
        if i == -1:
            return True

    if(actions(state)):
        return False
    return True


def utlity(state: list) -> int:
    # given that the state passed in is terminal
    board = state_to_board(state)

    for i in board[0]:
        if i == 1:
            return 1

    for i in board[2]:
        if i == -1:
            return -1

    return 0


def actions(state: list) -> dict:
    # diagonally left = L, straight = S, diagonally right = R
    actions = {}
    board = state_to_board(state)

    if(to_move(state) == "white"):
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == 1:
                    temp = []
                    if i-1 >= 0 and j-1 >= 0:
                        if board[i-1][j-1] == -1:
                            temp.append("L" + str(i-1) + str(j-1))
                    if i-1 >= 0:
                        if board[i-1][j] == 0:
                            temp.append("S" + str(i-1) + str(j))
                    if i-1 >= 0 and j + 1 >= 0 and j + 1 < 3:
                        if board[i-1][j+1] == -1:
                            temp.append("R" + str(i-1) + str(j+1))
                    if temp:
                        actions["W" + str(i) + str(j)] = temp
    else:
        for i in range(len(board)):
            for j in range(len(board)):
                if board[i][j] == -1:
                    temp = []
                    if i + 1 < 3 and j - 1 >= 0:
                        if board[i+1][j-1] == 1:
                            temp.append("L" + str(i+1) + str(j-1))
                    if i+1 < 3:
                        if board[i+1][j] == 0:
                            temp.append("S" + str(i+1) + str(j))
                    if i+1 < 3 and j+1 < 3:
                        if board[i+1][j+1] == 1:
                            temp.append("R" + str(i+1) + str(j+1))
                    if temp:
                        actions["B" + str(i) + str(j)] = temp

    return actions


def result(state: list, position: str, action: str) -> list:
    board = state_to_board(state)
    position_i = int(position[1])
    position_j = int(position[2])
    action_i = int(action[1])
    action_j = int(action[2])

    board[action_i][action_j] = board[position_i][position_j]
    board[position_i][position_j] = 0

    return board_to_state(board, 0 if state[0] == 1 else 1)

def minimax(state: list) -> list:
    player = to_move(state)
    vm_pair = max_value(state)
    move = vm_pair[1]
    return move


def max_value(state: list) -> list:
    if is_terminal(state):
        return [utlity(state), None]

    value = -5  # could be any value < -1
    for position in actions(state):
        for action in actions(state)[position]:
            vm_pair = min_value(result(state, position, action))
            if vm_pair[0] > value:
                value = vm_pair[0]
                move = [position, action]
    return [value, move]


def min_value(state: list) -> list:
    if is_terminal(state):
        return [utlity(state), None]

    value = 5  # could be any value > 1
    for position in actions(state):
        for action in actions(state)[position]:
            vm_pair = min_value(result(state, position, action))
            if vm_pair[0] < value:
                value = vm_pair[0]
                move = [position, action]
    return [value, move]


def policy_table_wrapper(init_state: list) -> list:
    policytable = []
    return policy_table(init_state, policytable)


def create_move(state: list, position: str) -> list:
    board = state_to_board(state.copy())

    position_i = int(position[1])
    position_j = int(position[2])

    for i in range(len(board)):
        for j in range(len(board)):
            board[i][j] = 0

    board[position_i][position_j] = 1
    return board_to_state(board, state[0])[1:]


def policy_table(state: list, table: list) -> list:
    if is_terminal(state):
        return table

    move = minimax(state)
    new_state = result(state, move[0], move[1])
    table.append([state, create_move(state, move[0])])
    return policy_table(new_state, table)

def classify_wrapper(network: Network, input: list) -> list:
    counter = 0
    return classify(network, transpose(input), counter)


def transpose(state: list) -> list:
    ns = []
    for i in range(len(state)):
        temp = []
        for j in range(1):
            temp.append(state[i])
        ns.append(temp)
    return ns


def classify(network: Network, input: list, counter: int) -> list:
    layers = network.print_layers()
    if counter == len(layers):
        return input
    product = layers[counter].multiply_weight(input)
    updated_product = layers[counter].add_bias(product)
    # can use relu_activate here instead as well
    output = layers[counter].sigmoid_activate(updated_product)
    counter += 1
    return classify(network, output, counter)

def update_weights_wrapper(network: Network, output: list):
    return update_weights(network, transpose(output))


def update_weights(network: Network, output: list) -> None:
    # essentially back propagate
    counter = len(network.print_layers()) - 1
    formulate_error(network, output, counter)
    adjust_attributes(network, counter)


def formulate_error(network: Network, output: list, counter: int) -> None:
    if counter < 0:
        return
    layer = network.print_layers()[counter]
    if layer.is_output_layer():
        network.set_error_outer_layer(layer, output)
    else:
        network.set_error_hidden_layer(
            layer, network.print_layers()[counter + 1])
    network.set_error_delta(layer)
    counter -= 1
    return formulate_error(network, output, counter)


def adjust_attributes(network: Network, counter: int):
    if counter < 0:
        return
    layer = network.print_layers()[counter]
    layer.set_adj_amount()
    layer.adjust_weights()
    layer.adjust_bias()
    counter -= 1
    return adjust_attributes(network, counter)

#test
initial_state = [0, -1, -1, -1, 0, 0, 0, 1, 1, 1]

network = Network()
network.add_layer(True, False)
network.add_layer(False, True)

policytable = policy_table_wrapper(initial_state)

# randomly inputting inputs/outputs from policy table
i = 100000
while(i != 0):
    entry = random.choice(policytable)
    classify_wrapper(network, entry[0])
    update_weights_wrapper(network, entry[1])
    print(entry[1], classify_wrapper(network, entry[0]))
    i -= 1
# systematically feeding in inputs/output from policy table
j = 100000
while(i != 0):
    for entry in policytable:
        classify_wrapper(network, entry[0])
        update_weights_wrapper(network, entry[1])
        print(entry[1], classify_wrapper(network, entry[0]))
