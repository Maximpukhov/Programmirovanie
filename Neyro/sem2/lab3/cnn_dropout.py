import numpy as np
import urllib.request
import gzip
import os
import struct
import matplotlib.pyplot as plt
import pickle
from datetime import datetime

class Dropout:
    def __init__(self, dropout_rate=0.5):
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x):
        if self.training:
            self.mask = (np.random.random(x.shape) > self.dropout_rate).astype(float)
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x
    
    def backward(self, doutput):
        if self.training:
            return doutput * self.mask / (1 - self.dropout_rate)
        else:
            return doutput

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

class CNNWithDropout:
    def __init__(self, learning_rate=0.01, dropout_rate=0.5):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        self.layers = [
            ConvLayer(1, 8, kernel_size=3, padding=1),
            ReLU(),
            MaxPoolLayer(2, stride=2),
            ConvLayer(8, 16, kernel_size=3, padding=1),
            ReLU(),
            MaxPoolLayer(2, stride=2),
            FlattenLayer(),
            Dropout(dropout_rate),
            DenseLayer(7*7*16, 128),
            ReLU(),
            Dropout(dropout_rate),
            DenseLayer(128, 10),
            Softmax()
        ]
    
    def set_training(self, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
    
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
            elif isinstance(layer, (ReLU, Softmax, MaxPoolLayer, FlattenLayer, Dropout)):
                doutput = layer.backward(doutput)
        return doutput
    
    def compute_loss(self, y_pred, y_true):
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[np.arange(m), np.argmax(y_true, axis=1)])
        return np.sum(log_likelihood) / m
    
    def predict(self, x):
        self.set_training(False)
        probabilities = self.forward(x)
        self.set_training(True)
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

def split_validation_data(X, y, validation_ratio=0.2):
    num_validation = int(X.shape[0] * validation_ratio)
    indices = np.random.permutation(X.shape[0])
    train_indices = indices[num_validation:]
    val_indices = indices[:num_validation]
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    return X_train, y_train, X_val, y_val

def save_checkpoint(model, epoch, metrics, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state': {f'layer_{i}': layer for i, layer in enumerate(model.layers) 
                       if isinstance(layer, (ConvLayer, DenseLayer))},
        'metrics': metrics,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved: {filename}")

def train_with_dropout_and_checkpoints():
    X_train_full, y_train_full, X_test, y_test = load_mnist()
    X_train, y_train, X_val, y_val = split_validation_data(X_train_full, y_train_full, 0.2)
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Validation samples: {X_val.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")
    
    cnn = CNNWithDropout(learning_rate=0.01, dropout_rate=0.5)
    epochs = 20
    batch_size = 64
    checkpoint_epochs = [5, 10, 15, 20]
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'test_acc': []
    }
    
    print("\nНачало обучения CNN с Dropout...")
    print("Эпоха\tTrain Loss\tTrain Acc\tVal Acc\tTest Acc")
    print("-" * 60)
    
    for epoch in range(1, epochs + 1):
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
        val_acc = cnn.accuracy(X_val[:1000], y_val[:1000])
        test_acc = cnn.accuracy(X_test[:1000], y_test[:1000])
        
        history['train_loss'].append(avg_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['test_acc'].append(test_acc)
        
        print(f"{epoch}\t{avg_loss:.4f}\t\t{train_acc:.4f}\t\t{val_acc:.4f}\t{test_acc:.4f}")
        
        if epoch in checkpoint_epochs:
            metrics = {
                'train_loss': avg_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            }
            save_checkpoint(cnn, epoch, metrics, f'checkpoint_epoch_{epoch}.pkl')
    
    return cnn, history

def run_autotests():
    print("\n" + "="*50)
    print("ЗАПУСК АВТОТЕСТОВ")
    print("="*50)
    
    X_train_full, y_train_full, X_test, y_test = load_mnist()
    test_sizes = [1000, 5000, 10000, 30000]
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    results = {}
    
    for size in test_sizes:
        print(f"\nТест с размером обучающей выборки: {size}")
        print("-" * 40)
        
        X_train_small = X_train_full[:size]
        y_train_small = y_train_full[:size]
        X_train, y_train, X_val, y_val = split_validation_data(X_train_small, y_train_small, 0.2)
        
        for dropout_rate in dropout_rates:
            print(f"Dropout rate: {dropout_rate}")
            cnn = CNNWithDropout(learning_rate=0.01, dropout_rate=dropout_rate)
            
            for epoch in range(5):
                permutation = np.random.permutation(X_train.shape[0])
                X_shuffled = X_train[permutation]
                y_shuffled = y_train[permutation]
                
                for i in range(0, X_train.shape[0], 64):
                    X_batch = X_shuffled[i:i+64]
                    y_batch = y_shuffled[i:i+64]
                    y_pred = cnn.forward(X_batch)
                    doutput = y_pred - y_batch
                    cnn.backward(doutput)
            
            train_acc = cnn.accuracy(X_train[:500], y_train[:500])
            val_acc = cnn.accuracy(X_val[:500], y_val[:500])
            test_acc = cnn.accuracy(X_test[:500], y_test[:500])
            
            results[(size, dropout_rate)] = {
                'train_acc': train_acc,
                'val_acc': val_acc,
                'test_acc': test_acc
            }
            
            print(f"  Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")
    
    return results

def plot_results(history, autotest_results):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    ax1.plot(history['train_loss'])
    ax1.set_title('Функция потерь с Dropout')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Loss')
    ax1.grid(True)
    
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.plot(history['test_acc'], label='Test Accuracy')
    ax2.set_title('Точность с Dropout')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    test_size = 10000
    dropout_rates = [0.0, 0.3, 0.5, 0.7]
    val_accs = [autotest_results[(test_size, dr)]['val_acc'] for dr in dropout_rates]
    test_accs = [autotest_results[(test_size, dr)]['test_acc'] for dr in dropout_rates]
    
    ax3.bar(np.arange(len(dropout_rates)) - 0.2, val_accs, 0.4, label='Validation')
    ax3.bar(np.arange(len(dropout_rates)) + 0.2, test_accs, 0.4, label='Test')
    ax3.set_title('Влияние Dropout на точность')
    ax3.set_xlabel('Dropout Rate')
    ax3.set_ylabel('Accuracy')
    ax3.set_xticks(range(len(dropout_rates)))
    ax3.set_xticklabels(dropout_rates)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    dropout_rate = 0.5
    sizes = [1000, 5000, 10000, 30000]
    val_accs_size = [autotest_results[(size, dropout_rate)]['val_acc'] for size in sizes]
    
    ax4.plot(sizes, val_accs_size, 'o-')
    ax4.set_title('Влияние размера данных на точность')
    ax4.set_xlabel('Размер обучающей выборки')
    ax4.set_ylabel('Validation Accuracy')
    ax4.grid(True)
    
    plt.tight_layout()
    plt.savefig('dropout_results.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    model, history = train_with_dropout_and_checkpoints()
    autotest_results = run_autotests()
    plot_results(history, autotest_results)
    
    print("\n" + "="*50)
    print("ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("="*50)
    print("Сохраненные чекпоинты:")
    for epoch in [5, 10, 15, 20]:
        print(f"  - checkpoint_
