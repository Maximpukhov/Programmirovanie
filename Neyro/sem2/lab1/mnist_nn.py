import numpy as np
import os
import urllib.request
import gzip
import pickle
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.W1 = np.random.randn(input_size, hidden_size1) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size1))
        self.W2 = np.random.randn(hidden_size1, hidden_size2) * np.sqrt(2.0 / hidden_size1)
        self.b2 = np.zeros((1, hidden_size2))
        self.W3 = np.random.randn(hidden_size2, output_size) * np.sqrt(2.0 / hidden_size2)
        self.b3 = np.zeros((1, output_size))

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return (x > 0).astype(float)

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)

        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.relu(self.z2)

        self.z3 = np.dot(self.a2, self.W3) + self.b3
        self.a3 = self.softmax(self.z3)
        return self.a3

    def backward(self, X, y, output):
        m = X.shape[0]

        dz3 = output - y
        dW3 = np.dot(self.a2.T, dz3) / m
        db3 = np.sum(dz3, axis=0, keepdims=True) / m

        dz2 = np.dot(dz3, self.W3.T) * self.relu_derivative(self.z2)
        dW2 = np.dot(self.a1.T, dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = np.dot(X.T, dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W3 -= self.learning_rate * dW3
        self.b3 -= self.learning_rate * db3
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1

    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), np.argmax(y_true, axis=1)] + 1e-8)
        loss = np.sum(log_likelihood) / m
        return loss

    def compute_accuracy(self, y_true, y_pred):
        predictions = np.argmax(y_pred, axis=1)
        labels = np.argmax(y_true, axis=1)
        accuracy = np.mean(predictions == labels)
        return accuracy

class MNISTLoader:
    def __init__(self):
        self.data_files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }

    def download_mnist(self):
        try:
            from tensorflow.keras.datasets import mnist
            (X_train, y_train), (X_test, y_test) = mnist.load_data()

            X_train = X_train.reshape(-1, 784).astype(np.float32) / 255.0
            X_test = X_test.reshape(-1, 784).astype(np.float32) / 255.0

            y_train_onehot = np.zeros((y_train.shape[0], 10))
            y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

            y_test_onehot = np.zeros((y_test.shape[0], 10))
            y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

            return X_train, y_train_onehot, X_test, y_test_onehot

        except ImportError:
            return self.download_mnist_alternative()

    def download_mnist_alternative(self):
        mnist_url = "https://github.com/mnielsen/neural-networks-and-deep-learning/raw/master/data/mnist.pkl.gz"
        mnist_file = "mnist.pkl.gz"

        if not os.path.exists(mnist_file):
            print("Downloading MNIST from alternative source...")
            urllib.request.urlretrieve(mnist_url, mnist_file)
        
        with gzip.open(mnist_file, 'rb') as f:
            train_data, val_data, test_data = pickle.load(f, encoding='latin1')

        X_train, y_train = train_data
        X_test, y_test = test_data

        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)

        y_train_onehot = np.zeros((y_train.shape[0], 10))
        y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

        y_test_onehot = np.zeros((y_test.shape[0], 10))
        y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

        return X_train, y_train_onehot, X_test, y_test_onehot

def plot_training_history(train_loss_history, train_accuracy_history, test_accuracy_history):
    epochs = range(1, len(train_loss_history) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot(epochs, train_loss_history, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(epochs, train_accuracy_history, 'g-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, test_accuracy_history, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig('training_plots_2hidden.png', dpi=300, bbox_inches='tight')
    plt.show()

def train_neural_network():
    print("Loading MNIST dataset...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.download_mnist()

    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")

    input_size = 784
    hidden_size1 = 128
    hidden_size2 = 64
    output_size = 10
    learning_rate = 0.1

    nn = NeuralNetwork(input_size, hidden_size1, hidden_size2, output_size, learning_rate)

    epochs = 10
    batch_size = 64

    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []

    print("\nStarting training...")
    for epoch in range(epochs):
        indices = np.random.permutation(X_train.shape[0])
        X_shuffled = X_train[indices]
        y_shuffled = y_train[indices]

        total_loss = 0
        total_accuracy = 0
        num_batches = 0

        for i in range(0, X_train.shape[0], batch_size):
            end_idx = min(i + batch_size, X_train.shape[0])
            X_batch = X_shuffled[i:end_idx]
            y_batch = y_shuffled[i:end_idx]

            output = nn.forward(X_batch)

            loss = nn.compute_loss(y_batch, output)
            accuracy = nn.compute_accuracy(y_batch, output)

            nn.backward(X_batch, y_batch, output)

            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        test_output = nn.forward(X_test)
        test_accuracy = nn.compute_accuracy(y_test, test_output)

        train_loss_history.append(avg_loss)
        train_accuracy_history.append(avg_accuracy)
        test_accuracy_history.append(test_accuracy)

        print(f"Epoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_loss:.4f}, Train Accuracy: {avg_accuracy:.4f}")
        print(f"  Test Accuracy: {test_accuracy:.4f}")

    final_output = nn.forward(X_test)
    final_accuracy = nn.compute_accuracy(y_test, final_output)
    final_loss = nn.compute_loss(y_test, final_output)

    print(f"\nFinal Results:")
    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_accuracy:.4f}")

    plot_training_history(train_loss_history, train_accuracy_history, test_accuracy_history)

    return nn, train_loss_history, train_accuracy_history, test_accuracy_history

if __name__ == "__main__":
    print("Neural Network Training on MNIST Dataset")
    print("=" * 50)

    try:
        trained_nn, train_loss, train_acc, test_acc = train_neural_network()

        print("\nTraining summary:")
        print(f"Final training accuracy: {train_acc[-1]:.4f}")
        print(f"Final test accuracy: {test_acc[-1]:.4f}")
        print(f"Best test accuracy: {max(test_acc):.4f}")

    except Exception as e:
        print(f"Error during training: {e}")
        print("Please make sure you have internet connection to download MNIST dataset")
