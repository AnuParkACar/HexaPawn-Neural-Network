# HexaPawn-Neural-Network
AI based solution for a 3 x 3 chess board

## Description

This ia Python project focused on creating an Artificial Intelligence (AI) for a 3x3 chess board. It's a demonstration of how the Minimax algorithm, along with some Machine Learning concepts, namely a neural network, can be applied to create an intelligent decision-making entity for board games. The AI uses a set of functions to process the game state and perform the optimal move.

The program consists of three main components: `main.py`, `network.py`, and `layer.py`.

### Main.py 

The `main.py` script includes a set of functions responsible for game mechanics, AI decision-making, and training processes. The functions handle game state transformations (`state_to_board`, `board_to_state`), determine available actions (`actions`), check game states (`is_terminal`), calculate utilities (`utility`), and more. It uses the Minimax algorithm with policy tables to determine the optimal moves. The Neural Network's weight updating process is also handled here.

### Network.py 

The `network.py` script defines the `Network` class, representing a Neural Network. The class includes methods to add layers, print layers, set error for layers, etc. This class is essential for creating and managing the neural network used for training the AI.

### Layer.py

The `Layer.py` script introduces the `Layer` class that represents a layer in the Neural Network. It consists of numerous methods for creating a layer, creating and retrieving a bias, multiplying weights, adding bias, performing activation functions (ReLU and Sigmoid), adjusting weights, and other utilities related to neural network layers.

Each script and function plays a pivotal role in creating the intelligent agent capable of making strategic decisions in the game. It's a showcase of combining various AI techniques and concepts to simulate intelligent behavior in a restricted environment.

This project is a great resource for individuals interested in game AI, machine learning, and Python programming. Contributions and enhancements to the project are welcomed.

## Instructions

Clone the repository, navigate to the project directory, and run the `main.py` script to start the game. Ensure Python 3.x is installed in your environment and all necessary libraries are present.
