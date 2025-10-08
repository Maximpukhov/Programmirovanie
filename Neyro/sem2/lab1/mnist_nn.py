import numpy as np
import struct
import gzip
import os
import urllib.request
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        # Инициализация весов
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def forward(self, X):
        # Прямой проход
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def compute_loss(self, y_pred, y_true):
        # Функция потерь: кросс-энтропия
        m = y_true.shape[0]
        log_likelihood = -np.log(y_pred[range(m), y_true.argmax(axis=1)])
        loss = np.sum(log_likelihood) / m
        return loss
    
    def backward(self, X, y_true, y_pred):
        m = X.shape[0]
        
        # Обратный проход
        dz2 = y_pred - y_true
        dW2 = (1/m) * np.dot(self.a1.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        dz1 = np.dot(dz2, self.W2.T) * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # Обновление весов
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def predict(self, X):
        probabilities = self.forward(X)
        return np.argmax(probabilities, axis=1)
    
    def accuracy(self, X, y):
        predictions = self.predict(X)
        true_labels = np.argmax(y, axis=1)
        return np.mean(predictions == true_labels)

class MNISTLoader:
    def __init__(self):
        self.url_base = 'http://yann.lecun.com/exdb/mnist/'
        self.files = {
            'train_images': 'train-images-idx3-ubyte.gz',
            'train_labels': 'train-labels-idx1-ubyte.gz',
            'test_images': 't10k-images-idx3-ubyte.gz',
            'test_labels': 't10k-labels-idx1-ubyte.gz'
        }
    
    def download_mnist(self):
        if not os.path.exists('data'):
            os.makedirs('data')
        
        for file_type, filename in self.files.items():
            filepath = f'data/{filename}'
            if not os.path.exists(filepath):
                print(f'Downloading {filename}...')
                urllib.request.urlretrieve(self.url_base + filename, filepath)
    
    def load_images(self, filename):
        with gzip.open(filename, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)
            return images / 255.0  # Нормализация
    
    def load_labels(self, filename):
        with gzip.open(filename, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            labels = np.frombuffer(f.read(), dtype=np.uint8)
            return labels
    
    def load_data(self):
        self.download_mnist()
        
        X_train = self.load_images('data/train-images-idx3-ubyte.gz')
        y_train = self.load_labels('data/train-labels-idx1-ubyte.gz')
        X_test = self.load_images('data/t10k-images-idx3-ubyte.gz')
        y_test = self.load_labels('data/t10k-images-idx1-ubyte.gz')
        
        # One-hot encoding
        encoder = OneHotEncoder(sparse_output=False, categories=[range(10)])
        y_train_onehot = encoder.fit_transform(y_train.reshape(-1, 1))
        y_test_onehot = encoder.transform(y_test.reshape(-1, 1))
        
        return X_train, y_train_onehot, X_test, y_test_onehot, y_train, y_test

def train_model():
    # Загрузка данных
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test, y_train_labels, y_test_labels = loader.load_data()
    
    # Создание и обучение модели
    input_size = 784  # 28x28 пикселей
    hidden_size = 128  # Оптимальный размер скрытого слоя
    output_size = 10   # 10 классов цифр
    learning_rate = 0.1
    epochs = 50
    batch_size = 64
    
    nn = NeuralNetwork(input_size, hidden_size, output_size, learning_rate)
    
    # История для графиков
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print("Начало обучения...")
    print("Эпоха\tTrain Loss\tTrain Acc\tTest Acc")
    print("-" * 50)
    
    for epoch in range(epochs):
        # Мини-батч обучение
        permutation = np.random.permutation(X_train.shape[0])
        X_train_shuffled = X_train[permutation]
        y_train_shuffled = y_train[permutation]
        
        epoch_loss = 0
        num_batches = 0
        
        for i in range(0, X_train.shape[0], batch_size):
            X_batch = X_train_shuffled[i:i+batch_size]
            y_batch = y_train_shuffled[i:i+batch_size]
            
            # Прямой проход
            y_pred = nn.forward(X_batch)
            
            # Вычисление потерь
            loss = nn.compute_loss(y_pred, y_batch)
            epoch_loss += loss
            num_batches += 1
            
            # Обратный проход
            nn.backward(X_batch, y_batch, y_pred)
        
        # Вычисление метрик
        avg_loss = epoch_loss / num_batches
        train_acc = nn.accuracy(X_train, y_train)
        test_acc = nn.accuracy(X_test, y_test)
        
        train_losses.append(avg_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        if epoch % 5 == 0:
            print(f"{epoch}\t{avg_loss:.4f}\t\t{train_acc:.4f}\t\t{test_acc:.4f}")
    
    # Финальные результаты
    final_train_acc = nn.accuracy(X_train, y_train)
    final_test_acc = nn.accuracy(X_test, y_test)
    
    print("-" * 50)
    print(f"Финальная точность на тренировочных данных: {final_train_acc:.4f}")
    print(f"Финальная точность на тестовых данных: {final_test_acc:.4f}")
    
    # Визуализация обучения
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Функция потерь во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Loss')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(test_accuracies, label='Test Accuracy')
    plt.title('Точность во время обучения')
    plt.xlabel('Эпоха')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Демонстрация нескольких предсказаний
    show_predictions(nn, X_test, y_test_labels, loader)

def show_predictions(model, X_test, y_test, loader):
    """Показать несколько примеров предсказаний"""
    indices = np.random.choice(len(X_test), 10, replace=False)
    
    plt.figure(figsize=(15, 6))
    for i, idx in enumerate(indices):
        plt.subplot(2, 5, i + 1)
        image = X_test[idx].reshape(28, 28)
        plt.imshow(image, cmap='gray')
        
        prediction = model.predict(X_test[idx:idx+1])[0]
        true_label = y_test[idx]
        
        color = 'green' if prediction == true_label else 'red'
        plt.title(f'True: {true_label}, Pred: {prediction}', color=color)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    train_model()
