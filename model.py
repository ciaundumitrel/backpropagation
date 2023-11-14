import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class Model:
    def __init__(self, hidden_layers_count=None, file=None, learning_rate=0.1, test_size=0.2):
        """
        :param hidden_layers_count:
        :param file:
        :param test_size:
        """
        tf.compat.v1.enable_eager_execution()

        self.model = None
        self.rows_count = None
        self.cols_count = None
        self.file = file
        self.data_size = None
        self.training_data = None
        self.testing_data = None
        self.hidden_layers_size = hidden_layers_count
        self.test_size = test_size
        self.learning_rate = learning_rate

    def process_data(self) -> None:
        """
        Process the given file
        :return: None
        """
        data = np.array([[0, 0, 0, 0, 0, 0, 0, 0]])  # Initialize data as None

        try:
            data = pd.read_csv(self.file, sep='\t', header=None, dtype=float)
        except pd.errors.ParserError as e:
            for row in pd.read_csv(self.file, header=None)[0]:
                row_values = [float(value.strip()) for value in row.split('\t') if value]

                if data is None:
                    data = np.array([row_values])
                else:
                    data = np.concatenate([data, np.array([row_values])])

        self.rows_count = data.shape[0]
        self.cols_count = data.shape[1]

        df = pd.DataFrame(data[1:])
        self.training_data, self.testing_data = train_test_split(df, test_size=self.test_size, random_state=50)

    def build_neural_network(self):
        input_dim = self.cols_count - 1

        self.model = tf.keras.Sequential()
        self.model.add(tf.keras.layers.Dense(self.hidden_layers_size, input_dim=input_dim, activation=self.custom_activation))
        self.model.add(tf.keras.layers.Dense(1, activation='linear'))

        self.model.compile(optimizer='adam', loss=self.custom_error_function)

    def extract_features_and_targets(self):
        return self.training_data.iloc[:, :7], self.training_data.iloc[:, 7:]

    def custom_activation(self, x):
        return tf.keras.activations.relu(x)

    def custom_activation_derivative(self, x):
        return tf.where(x > 0, 1.0, 0.0)

    def custom_error_function(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_true - y_pred))

    def train_neural_network(self):
        X_train, y_train = self.extract_features_and_targets()
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="logs", histogram_freq=1)

        self.model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2,
                       callbacks=[tensorboard_callback])

    def forward_propagation(self, input_data):
        # Convert input_data to a NumPy array if it's not already
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        # Forward propagation through the hidden layer
        hidden_layer_output = self.custom_activation(
            np.dot(input_data, self.model.layers[0].get_weights()[0]) + self.model.layers[0].get_weights()[1])

        # Forward propagation through the output layer
        output_layer_output = np.dot(hidden_layer_output, self.model.layers[1].get_weights()[0]) + \
                              self.model.layers[1].get_weights()[1]

        return output_layer_output, hidden_layer_output

    def backward_propagation(self, input_data, target):
        if not isinstance(input_data, np.ndarray):
            input_data = np.array(input_data)

        if not isinstance(target, np.ndarray):
            target = np.array(target)

        output_layer_output, hidden_layer_output = self.forward_propagation(input_data)

        output_error = output_layer_output - target

        output_weights_gradient = np.dot(hidden_layer_output, output_error)
        output_bias_gradient = np.sum(output_error, axis=0, keepdims=True)

        # Update the output layer weights and biases
        self.model.layers[1].set_weights([
            self.model.layers[1].get_weights()[0] - self.learning_rate * output_weights_gradient,
            self.model.layers[1].get_weights()[1] - self.learning_rate * output_bias_gradient
        ])

        # Compute the derivative of the loss with respect to the hidden layer output
        hidden_layer_error = np.dot(output_error, self.model.layers[1].get_weights()[0].T)

        # Compute the derivative of the hidden layer output with respect to its input
        hidden_layer_activation_derivative = self.custom_activation_derivative(
            np.dot(input_data, self.model.layers[0].get_weights()[0]) + self.model.layers[0].get_weights()[1])

        # Element-wise multiplication of the error and the derivative of the activation function
        hidden_layer_error *= hidden_layer_activation_derivative

        # Compute gradients for the hidden layer weights and biases
        hidden_weights_gradient = np.dot(input_data.T, hidden_layer_error)
        hidden_bias_gradient = np.sum(hidden_layer_error, axis=0, keepdims=True)

        # Update the hidden layer weights and biases
        self.model.layers[0].set_weights([
            self.model.layers[0].get_weights()[0] - self.learning_rate * hidden_weights_gradient,
            self.model.layers[0].get_weights()[1] - self.learning_rate * hidden_bias_gradient
        ])