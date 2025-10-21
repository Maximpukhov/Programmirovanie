import numpy as np
import urllib.request
import gzip
import os
import struct
import matplotlib.pyplot as plt

class ConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * 0.1
        self.biases = np.zeros(output_channels)
        self.dweights = None
        self.dbiases = None
        self.input = None
    
    def forward(self, x):
        self.input = x
        batch_size, input_channels, input_height, input_width = x.shape
        
        if self.padding > 0:
            x_padded = np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding), 
                                (self.padding, self.padding)), mode='constant')
        else:
            x_padded = x
        
        output_height = (input_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        output_width = (input_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        output = np.zeros((batch_size, self.output_channels, output_height, output_width))
        
        for b in range(batch_size):
            for out_ch in range(self.output_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        output[b, out_ch, i, j] = np.sum(patch * self.weights[out_ch]) + self.biases[out_ch]
        
        return output
    
    def backward(self, doutput, learning_rate):
        batch_size, output_channels, output_height, output_width = doutput.shape
        input_channels = self.input_channels
        
        dinput = np.zeros_like(self.input)
        self.dweights = np.zeros_like(self.weights)
        self.dbiases = np.zeros_like(self.biases)
        
        if self.padding > 0:
            input_padded = np.pad(self.input, ((0, 0), (0, 0), (self.padding, self.padding), 
                                            (self.padding, self.padding)), mode='constant')
            dinput_padded = np.pad(dinput, ((0, 0), (0, 0), (self.padding, self.padding), 
                                          (self.padding, self.padding)), mode='constant')
        else:
            input_padded = self.input
            dinput_padded = dinput
        
        for b in range(batch_size):
            for out_ch in range(output_channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.kernel_size
                        w_start = j * self.stride
                        w_end = w_start + self.kernel_size
                        patch = input_padded[b, :, h_start:h_end, w_start:w_end]
                        self.dweights[out_ch] += doutput[b, out_ch, i, j] * patch
                        self.dbiases[out_ch] += doutput[b, out_ch, i, j]
                        dinput_padded[b, :, h_start:h_end, w_start:w_end] += doutput[b, out_ch, i, j] * self.weights[out_ch]
        
        if self.padding > 0:
            dinput = dinput_padded[:, :, self.padding:-self.padding, self.padding:-self.padding]
        else:
            dinput = dinput_padded
        
        self.weights -= learning_rate * self.dweights / batch_size
        self.biases -= learning_rate * self.dbiases / batch_size
        
        return dinput

class MaxPoolLayer:
    def __init__(self, pool_size, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self.input = None
        self.mask = None
    
    def forward(self, x):
        self.input = x
        batch_size, channels, input_height, input_width = x.shape
        
        output_height = (input_height - self.pool_size) // self.stride + 1
        output_width = (input_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, output_height, output_width))
        self.mask = np.zeros_like(x)
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, i, j] = np.max(patch)
                        max_pos = np.unravel_index(np.argmax(patch), patch.shape)
                        self.mask[b, c, h_start + max_pos[0], w_start + max_pos[1]] = 1
        
        return output
    
    def backward(self, doutput):
        dinput = np.zeros_like(self.input)
        batch_size, channels, output_height, output_width = doutput.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for i in range(output_height):
                    for j in range(output_width):
                        h_start = i * self.stride
                        h_end = h_start + self.pool_size
                        w_start = j * self.stride
                        w_end = w_start + self.pool_size
                        dinput[b, c, h_start:h_end, w_start:w_end] += self.mask[b, c, h_start:h_end, w_start:w_end] * doutput[b, c, i, j]
        
        return dinput

class FlattenLayer:
    def __init__(self):
        self.input_shape = None
    
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, doutput):
        return doutput.reshape(self.input_shape)

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * 0.1
        self.biases = np.zeros(output_size)
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.biases
    
    def backward(self, doutput, learning_rate):
        batch_size = doutput.shape[0]
        dinput = np.dot(doutput, self.weights.T)
        dweights = np.dot(self.input.T, doutput)
        dbiases = np.sum(doutput, axis=0)
        self.weights -= learning_rate * dweights / batch_size
        self.biases -= learning_rate * dbiases / batch_size
        return dinput

class ReLU:
    def __init__(self):
        self.input = None
    
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, doutput):
        dinput = doutput.copy()
        dinput[self.input <= 0] = 0
        return dinput

class Softmax:
    def __init__(self):
        self.output = None
    
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, doutput):
        return doutput

class CNN:
    def __init__(self, learning_rate=0.01):
        self.layers = [
            ConvLayer(1, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPoolLayer(2, stride=2),
            ConvLayer(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPoolLayer(2, stride=2),
            FlattenLayer(),
            DenseLayer(7*7*16, 128),
            ReLU(),
            DenseLayer(128, 10),
            Softmax()
        ]
        self.learning_rate = learning_rate
    
    def forward(self, x):
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, doutput):
        for layer in reversed(self.layers):
            if isinstance(layer, (ConvLayer, DenseLayer)):
                doutput = layer.backward(doutput, self.learning_rate)
            elif isinstance(layer, (ReLU, Softmax, MaxPoolLayer, FlattenLayer)):
                doutput = layer.backward(doutput)
        return doutput
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), np.argmax(y_true, axis=1)])
        return np.sum(log_likelihood) / m
    
    def predict(self, x):
        probabilities = self.forward(x)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, x, y):
        predictions = self.predict(x)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

def load_mnist():
    files = {
        'train_images': 'train-images-idx3-ubyte.gz',
        'train_labels': 'train-labels-idx1-ubyte.gz',
        'test_images': 't10k-images-idx3-ubyte.gz',
        'test_labels': 't10k-labels-idx1-ubyte.gz'
    }
    
    for name, filename in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(f'http://yann.lecun.com/exdb/mnist/{filename}', filename)
    
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
            return images / 255.0
    
    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return np.eye(10)[labels]
    
    X_train = load_images('train-images-idx3-ubyte.gz')
    y_train = load_labels('train-labels-idx1-ubyte.gz')
    X_test = load_images('t10k-images-idx3-ubyte.gz') 
    y_test = load_labels('t10k-labels-idx1-ubyte.gz')
    
    return X_train, y_train, X_test, y_test

def train_cnn():
    X_train, y_train, X_test, y_test = load_mnist()
    cnn = CNN(learning_rate=0.01)
    epochs = 10
    batch_size = 32
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Начало обучения CNN...")
    print("Эпоха\tTrain Loss\tTrain Acc\tTest Acc")
    print("-" * 50)
    
    for epoch in range(epochs):
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            y_pred = cnn.forward(X_batch)
            loss = cnn.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            num_batches += 1
            doutput = y_pred - y_batch
            cnn.backward(doutput)
        
        avg_loss = epoch_loss / num_batches
        train_acc = cnn.accuracy(X_train[:1000], y_train[:1000])
        test_acc = cnn.accuracy(X_test[:1000], y_test[:1000])
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"{epoch}\t{avg_loss:.4f}\t\t{train_acc:.4f}\t\t{test_acc:.4f}")
    
    final_train_acc = cnn.accuracy(X_train[:2000], y_train[:2000])
    final_test_acc = cnn.accuracy(X_test[:2000], y_test[:2000])
    
    print("-" * 50)
    print(f"Финальная точность на тренировочных данных: {final_train_acc:.4f}")
    print(f"Финальная точность на тестовых данных: {final_test_acc:.4f}")
    
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Функция потерь CNN')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Точность CNN')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('cnn_training.png')
    plt.show()

if __name__ == "__main__":
    train_cnn()
