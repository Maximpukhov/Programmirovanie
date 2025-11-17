import numpy as np
import os
import urllib.request
import gzip
import pickle
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
import time
from IPython.display import clear_output

# Настройка для отображения графиков в Colab
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

class ReLU:
    def __init__(self):
        self.input = None
        
    def forward(self, x):
        self.input = x
        return np.maximum(0, x)
    
    def backward(self, d_output):
        return d_output * (self.input > 0)

class Softmax:
    def __init__(self):
        self.output = None
        
    def forward(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        self.output = exp_x / np.sum(exp_x, axis=1, keepdims=True)
        return self.output
    
    def backward(self, d_output):
        return d_output

class FlattenLayer:
    def __init__(self):
        self.input_shape = None
        
    def forward(self, x):
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)
    
    def backward(self, d_output):
        return d_output.reshape(self.input_shape)

class DenseLayer:
    def __init__(self, input_size, output_size):
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2.0 / input_size)
        self.bias = np.zeros((1, output_size))
        self.input = None
        
    def forward(self, x):
        self.input = x
        return np.dot(x, self.weights) + self.bias
    
    def backward(self, d_output, learning_rate):
        batch_size = self.input.shape[0]
        
        d_input = np.dot(d_output, self.weights.T)
        d_weights = np.dot(self.input.T, d_output)
        d_bias = np.sum(d_output, axis=0, keepdims=True)
        
        self.weights -= learning_rate * d_weights / batch_size
        self.bias -= learning_rate * d_bias / batch_size
        
        return d_input

class FastConvLayer:
    def __init__(self, input_channels, output_channels, kernel_size, stride=1, padding=0):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        scale = np.sqrt(2.0 / (input_channels * kernel_size * kernel_size))
        self.weights = np.random.randn(output_channels, input_channels, kernel_size, kernel_size) * scale
        self.bias = np.zeros(output_channels)
        
        self.input = None
        self.output = None
        
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
            for oc in range(self.output_channels):
                for ic in range(self.input_channels):
                    output[b, oc] += convolve2d(x_padded[b, ic], self.weights[oc, ic, ::-1, ::-1], 
                                              mode='valid')[::self.stride, ::self.stride]
                output[b, oc] += self.bias[oc]
        
        self.output = output
        return output
    
    def backward(self, d_output, learning_rate):
        batch_size = d_output.shape[0]
        d_input = np.zeros_like(self.input)
        d_weights = np.zeros_like(self.weights)
        d_bias = np.zeros_like(self.bias)
        
        d_bias = np.sum(d_output, axis=(0, 2, 3))
        
        for b in range(batch_size):
            for oc in range(self.output_channels):
                for ic in range(self.input_channels):
                    kernel_grad = convolve2d(self.input[b, ic], d_output[b, oc], mode='valid')
                    d_weights[oc, ic] += kernel_grad
                    
                    if self.padding > 0:
                        input_grad = convolve2d(d_output[b, oc], self.weights[oc, ic], mode='full')
                        d_input[b, ic] += input_grad[self.padding:-self.padding, self.padding:-self.padding]
                    else:
                        d_input[b, ic] += convolve2d(d_output[b, oc], self.weights[oc, ic], mode='full')
        
        self.weights -= learning_rate * d_weights / batch_size
        self.bias -= learning_rate * d_bias / batch_size
        
        return d_input

class FastMaxPoolLayer:
    def __init__(self, pool_size=2, stride=2):
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
                for oh in range(output_height):
                    for ow in range(output_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        window = x[b, c, h_start:h_end, w_start:w_end]
                        output[b, c, oh, ow] = np.max(window)
                        
                        max_pos = np.unravel_index(np.argmax(window), window.shape)
                        self.mask[b, c, h_start + max_pos[0], w_start + max_pos[1]] = 1
        
        return output
    
    def backward(self, d_output):
        d_input = np.zeros_like(self.input)
        batch_size, channels, output_height, output_width = d_output.shape
        
        for b in range(batch_size):
            for c in range(channels):
                for oh in range(output_height):
                    for ow in range(output_width):
                        h_start = oh * self.stride
                        w_start = ow * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        d_input[b, c, h_start:h_end, w_start:w_end] += (
                            d_output[b, c, oh, ow] * 
                            (self.input[b, c, h_start:h_end, w_start:w_end] == 
                             np.max(self.input[b, c, h_start:h_end, w_start:w_end]))
                        )
        
        return d_input

class FastCNN:
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        self.conv1 = FastConvLayer(1, 16, 3, padding=1)
        self.relu1 = ReLU()
        self.pool1 = FastMaxPoolLayer(2, 2)
        
        self.conv2 = FastConvLayer(16, 32, 3, padding=1)
        self.relu2 = ReLU()
        self.pool2 = FastMaxPoolLayer(2, 2)
        
        self.flatten = FlattenLayer()
        self.fc1 = DenseLayer(7*7*32, 64)
        self.relu3 = ReLU()
        self.fc2 = DenseLayer(64, 10)
        self.softmax = Softmax()
        
    def forward(self, x):
        if len(x.shape) == 2:
            x = x.reshape(x.shape[0], 1, 28, 28)
            
        x = self.conv1.forward(x)
        x = self.relu1.forward(x)
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = self.relu2.forward(x)
        x = self.pool2.forward(x)
        
        x = self.flatten.forward(x)
        x = self.fc1.forward(x)
        x = self.relu3.forward(x)
        x = self.fc2.forward(x)
        x = self.softmax.forward(x)
        
        return x
    
    def backward(self, x, y_true, output):
        batch_size = x.shape[0]
        
        d_output = output - y_true
        
        d_output = self.fc2.backward(d_output, self.learning_rate)
        d_output = self.relu3.backward(d_output)
        d_output = self.fc1.backward(d_output, self.learning_rate)
        d_output = self.flatten.backward(d_output)
        d_output = self.pool2.backward(d_output)
        d_output = self.relu2.backward(d_output)
        d_output = self.conv2.backward(d_output, self.learning_rate)
        d_output = self.pool1.backward(d_output)
        d_output = self.relu1.backward(d_output)
        d_output = self.conv1.backward(d_output, self.learning_rate)
    
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

            X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0
            X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

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

        X_train = X_train.reshape(-1, 1, 28, 28).astype(np.float32)
        X_test = X_test.reshape(-1, 1, 28, 28).astype(np.float32)

        y_train_onehot = np.zeros((y_train.shape[0], 10))
        y_train_onehot[np.arange(y_train.shape[0]), y_train] = 1

        y_test_onehot = np.zeros((y_test.shape[0], 10))
        y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

        return X_train, y_train_onehot, X_test, y_test_onehot

def visualize_sample_images(X_train, y_train):
    """Визуализация примеров изображений из датасета"""
    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for i in range(10):
        row, col = i // 5, i % 5
        # Находим первое изображение с меткой i
        idx = np.where(np.argmax(y_train, axis=1) == i)[0][0]
        axes[row, col].imshow(X_train[idx, 0], cmap='gray')
        axes[row, col].set_title(f'Label: {i}')
        axes[row, col].axis('off')
    plt.suptitle('Примеры изображений из MNIST датасета', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def visualize_feature_maps(model, sample_image):
    """Визуализация feature maps после сверточных слоев"""
    # Прямой проход до первого сверточного слоя
    conv1_output = model.conv1.forward(sample_image.reshape(1, 1, 28, 28))
    relu1_output = model.relu1.forward(conv1_output)
    
    # Прямой проход до второго сверточного слоя
    pool1_output = model.pool1.forward(relu1_output)
    conv2_output = model.conv2.forward(pool1_output)
    relu2_output = model.relu2.forward(conv2_output)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Визуализация feature maps из первого сверточного слоя
    axes[0, 0].imshow(conv1_output[0, 0], cmap='viridis')
    axes[0, 0].set_title('Conv1 - First Feature Map')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(conv1_output[0, 15], cmap='viridis')  # Последний фильтр
    axes[0, 1].set_title('Conv1 - Last Feature Map')
    axes[0, 1].axis('off')
    
    # Визуализация feature maps из второго сверточного слоя
    axes[1, 0].imshow(conv2_output[0, 0], cmap='viridis')
    axes[1, 0].set_title('Conv2 - First Feature Map')
    axes[1, 0].axis('off')
    
    axes[1, 1].imshow(conv2_output[0, 31], cmap='viridis')  # Последний фильтр
    axes[1, 1].set_title('Conv2 - Last Feature Map')
    axes[1, 1].axis('off')
    
    plt.suptitle('Feature Maps после сверточных слоев', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_training_progress(loss_history, accuracy_history, epoch, total_epochs):
    """Обновляемый график прогресса обучения"""
    clear_output(wait=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs_range = range(1, len(loss_history) + 1)
    
    ax1.plot(epochs_range, loss_history, 'b-', linewidth=2, marker='o')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(epochs_range)
    
    ax2.plot(epochs_range, accuracy_history, 'r-', linewidth=2, marker='o')
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.set_xticks(epochs_range)
    ax2.set_ylim(0, 1)
    
    plt.suptitle(f'Прогресс обучения (Эпоха {epoch}/{total_epochs})', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def plot_final_results(train_loss_history, train_accuracy_history, test_accuracy_history):
    """Финальные графики результатов обучения"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    epochs = range(1, len(train_loss_history) + 1)
    
    # График потерь
    ax1.plot(epochs, train_loss_history, 'b-', linewidth=2, label='Training Loss')
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # График точности
    ax2.plot(epochs, train_accuracy_history, 'g-', linewidth=2, label='Train Accuracy')
    ax2.plot(epochs, test_accuracy_history, 'r-', linewidth=2, label='Test Accuracy')
    ax2.set_title('Accuracy', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_ylim(0, 1)
    
    # График улучшения точности
    improvement = [test_accuracy_history[i] - train_accuracy_history[i] for i in range(len(test_accuracy_history))]
    ax3.bar(epochs, improvement, color='purple', alpha=0.7)
    ax3.set_title('Разница Test - Train Accuracy', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Accuracy Difference')
    ax3.grid(True, alpha=0.3)
    
    # Сравнение финальных результатов
    final_metrics = [train_accuracy_history[-1], test_accuracy_history[-1], max(test_accuracy_history)]
    metrics_labels = ['Final Train', 'Final Test', 'Best Test']
    colors = ['lightblue', 'lightcoral', 'lightgreen']
    
    ax4.bar(metrics_labels, final_metrics, color=colors, alpha=0.7)
    ax4.set_title('Финальные результаты', fontsize=14, fontweight='bold')
    ax4.set_ylabel('Accuracy')
    ax4.set_ylim(0, 1)
    for i, v in enumerate(final_metrics):
        ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('Результаты обучения CNN на MNIST', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cnn_training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def visualize_predictions(model, X_test, y_test, num_samples=10):
    """Визуализация предсказаний модели"""
    indices = np.random.choice(len(X_test), num_samples, replace=False)
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    axes = axes.ravel()
    
    correct = 0
    for i, idx in enumerate(indices):
        image = X_test[idx:idx+1]
        true_label = np.argmax(y_test[idx])
        
        prediction = model.forward(image)
        predicted_label = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        axes[i].imshow(image[0, 0], cmap='gray')
        color = 'green' if predicted_label == true_label else 'red'
        axes[i].set_title(f'True: {true_label}\nPred: {predicted_label}\nConf: {confidence:.2f}', 
                         color=color, fontsize=10)
        axes[i].axis('off')
        
        if predicted_label == true_label:
            correct += 1
    
    accuracy = correct / num_samples
    plt.suptitle(f'Примеры предсказаний CNN (Accuracy: {accuracy:.1%})', 
                fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return accuracy

def train_fast_cnn():
    print("🚀 Загрузка MNIST датасета...")
    loader = MNISTLoader()
    X_train, y_train, X_test, y_test = loader.download_mnist()

    print(f"📊 Обучающая выборка: {X_train.shape} примеров")
    print(f"📊 Тестовая выборка: {X_test.shape} примеров")
    
    # Визуализация примеров изображений
    print("\n📷 Визуализация примеров изображений...")
    visualize_sample_images(X_train, y_train)

    learning_rate = 0.01
    cnn = FastCNN(learning_rate)

    epochs = 3
    batch_size = 256

    train_loss_history = []
    train_accuracy_history = []
    test_accuracy_history = []
    
    print(f"\n🎯 Начало обучения FAST CNN...")
    print(f"📈 Параметры: {epochs} эпох, batch_size={batch_size}, learning_rate={learning_rate}")
    print("⏳ Ожидаемое время: 3-5 минут\n")

    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_start = time.time()
        
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

            output = cnn.forward(X_batch)

            loss = cnn.compute_loss(y_batch, output)
            accuracy = cnn.compute_accuracy(y_batch, output)

            cnn.backward(X_batch, y_batch, output)

            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        avg_loss = total_loss / num_batches
        avg_accuracy = total_accuracy / num_batches

        test_output = cnn.forward(X_test)
        test_accuracy = cnn.compute_accuracy(y_test, test_output)

        train_loss_history.append(avg_loss)
        train_accuracy_history.append(avg_accuracy)
        test_accuracy_history.append(test_accuracy)
        
        epoch_time = time.time() - epoch_start

        print(f"✅ Эпоха {epoch+1}/{epochs} завершена за {epoch_time:.1f} сек:")
        print(f"   📉 Потери: {avg_loss:.4f}")
        print(f"   🎯 Точность на обучении: {avg_accuracy:.4f}")
        print(f"   🎯 Точность на тесте: {test_accuracy:.4f}")
        print("-" * 50)
        
        # Обновление графика прогресса
        plot_training_progress(train_loss_history, train_accuracy_history, epoch+1, epochs)

    total_time = time.time() - start_time
    
    # Финальная оценка
    final_output = cnn.forward(X_test)
    final_accuracy = cnn.compute_accuracy(y_test, final_output)
    final_loss = cnn.compute_loss(y_test, final_output)

    print(f"\n🎉 Обучение завершено за {total_time:.1f} секунд!")
    print(f"📊 Финальные результаты:")
    print(f"   📉 Потери на тесте: {final_loss:.4f}")
    print(f"   🎯 Точность на тесте: {final_accuracy:.4f}")

    # Визуализация feature maps
    print("\n🔍 Визуализация feature maps...")
    sample_idx = np.random.randint(len(X_test))
    visualize_feature_maps(cnn, X_test[sample_idx])
    
    # Финальные графики
    print("\n📈 Построение финальных графиков...")
    plot_final_results(train_loss_history, train_accuracy_history, test_accuracy_history)
    
    # Визуализация предсказаний
    print("\n🔮 Визуализация предсказаний модели...")
    visualize_predictions(cnn, X_test, y_test)

    return cnn, train_loss_history, train_accuracy_history, test_accuracy_history

if __name__ == "__main__":
    print("🧠 FAST CNN Обучение на MNIST Dataset")
    print("=" * 60)
    
    try:
        trained_cnn, train_loss, train_acc, test_acc = train_fast_cnn()

        print("\n📋 Итоги обучения:")
        print(f"   ✅ Финальная точность на обучении: {train_acc[-1]:.4f}")
        print(f"   ✅ Финальная точность на тесте: {test_acc[-1]:.4f}")
        print(f"   🏆 Лучшая точность на тесте: {max(test_acc):.4f}")
        print(f"   📈 Улучшение за обучение: {test_acc[-1] - test_acc[0]:.4f}")

    except Exception as e:
        print(f"❌ Ошибка при обучении: {e}")
        import traceback
        traceback.print_exc()
