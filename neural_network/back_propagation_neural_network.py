import numpy as np
from matplotlib import pyplot as plt


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    """Derivative of the sigmoid function."""
    return x * (1 - x)


class DenseLayer:
    """A dense layer for the BP neural network."""

    def __init__(self, units: int, activation=sigmoid, learning_rate: float = 0.3, is_input_layer: bool = False):
        self.units = units
        self.weight = None
        self.bias = None
        self.activation = activation
        self.learning_rate = learning_rate
        self.is_input_layer = is_input_layer
        self.output = None
        self.xdata = None

    def initializer(self, back_units: int):
        """Initialize weights and biases."""
        rng = np.random.default_rng()
        self.weight = rng.normal(0, 0.5, (self.units, back_units))
        self.bias = rng.normal(0, 0.5, self.units).reshape(-1, 1)

    def forward_propagation(self, xdata: np.ndarray) -> np.ndarray:
        """Perform forward propagation."""
        self.xdata = xdata
        if self.is_input_layer:
            self.output = xdata
        else:
            z = np.dot(self.weight, self.xdata) + self.bias
            self.output = self.activation(z)
        return self.output

    def back_propagation(self, gradient: np.ndarray) -> np.ndarray:
        """Perform back propagation."""
        gradient_activation = self.activation(self.output)
        gradient = gradient * sigmoid_derivative(gradient_activation)

        gradient_weight = np.dot(gradient, self.xdata.T)
        gradient_bias = gradient

        self.weight -= self.learning_rate * gradient_weight
        self.bias -= self.learning_rate * gradient_bias.mean(axis=1, keepdims=True)

        return np.dot(self.weight.T, gradient)


class LossFunction:
    """Base class for loss functions."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate loss."""
        raise NotImplementedError

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Calculate gradient of the loss."""
        raise NotImplementedError


class MeanSquaredError(LossFunction):
    """Mean Squared Error loss function."""

    def calculate(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.mean(np.square(y_true - y_pred))

    def gradient(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size


class BPNN:
    """Back Propagation Neural Network model."""

    def __init__(self):
        self.layers = []
        self.loss_function = MeanSquaredError()
        self.train_mse = []
        self.fig_loss, self.ax_loss = plt.subplots()

    def add_layer(self, layer: DenseLayer):
        """Add a layer to the network."""
        self.layers.append(layer)

    def build(self):
        """Initialize layers and weights."""
        for i, layer in enumerate(self.layers):
            if i > 0:
                layer.initializer(self.layers[i - 1].units)
            layer.is_input_layer = (i == 0)

    def summary(self):
        """Print a summary of the network architecture."""
        for i, layer in enumerate(self.layers):
            print(f"Layer {i}:")
            print(f"  Weights shape: {layer.weight.shape}")
            print(f"  Bias shape: {layer.bias.shape}")

    def train(self, xdata: np.ndarray, ydata: np.ndarray, train_round: int, accuracy: float):
        """Train the network."""
        for _ in range(train_round):
            total_loss = 0
            for row in range(xdata.shape[0]):
                _xdata = xdata[row, :].reshape(-1, 1)
                _ydata = ydata[row, :].reshape(-1, 1)

                # Forward propagation
                for layer in self.layers:
                    _xdata = layer.forward_propagation(_xdata)

                # Calculate loss and gradient
                loss = self.loss_function.calculate(_ydata, _xdata)
                total_loss += loss
                gradient = self.loss_function.gradient(_ydata, _xdata)

                # Back propagation
                for layer in reversed(self.layers):
                    gradient = layer.back_propagation(gradient)

            mse = total_loss / xdata.shape[0]
            self.train_mse.append(mse)
            self.plot_loss()

            if mse < accuracy:
                print("Reached target accuracy.")
                return mse
        return None

    def plot_loss(self):
        """Plot the training loss."""
        self.ax_loss.clear()
        self.ax_loss.plot(self.train_mse, "r-")
        self.ax_loss.set_xlabel("Epochs")
        self.ax_loss.set_ylabel("Mean Squared Error")
        plt.pause(0.1)


def example():
    rng = np.random.default_rng()
    x = rng.normal(size=(10, 10))
    y = np.random.rand(10, 2)  # Random target values

    model = BPNN()
    model.add_layer(DenseLayer(10))
    model.add_layer(DenseLayer(20))
    model.add_layer(DenseLayer(30))
    model.add_layer(DenseLayer(2))
    model.build()
    model.summary()
    model.train(xdata=x, ydata=y, train_round=100, accuracy=0.01)


if __name__ == "__main__":
    plt.ion()  # Interactive mode for live plotting
    example()
    plt.show()  # Keep the plot open after training
