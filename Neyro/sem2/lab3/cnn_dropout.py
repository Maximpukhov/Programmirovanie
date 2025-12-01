import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime
from google.colab import files
import warnings
warnings.filterwarnings('ignore')

# Установка русского шрифта для matplotlib
plt.rcParams['font.family'] = 'DejaVu Sans'

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

class SimpleDenseLayer:
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

class SimpleCNN:
    def __init__(self, learning_rate=0.01, dropout_rate=0.5):
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        # Очень простая архитектура - только полносвязные слои
        self.layers = [
            SimpleDenseLayer(28*28, 64),  # Прямо из 28x28 в 64 нейрона
            ReLU(),
            Dropout(dropout_rate),
            SimpleDenseLayer(64, 32),
            ReLU(),
            Dropout(dropout_rate),
            SimpleDenseLayer(32, 10),
            Softmax()
        ]
    
    def set_training(self, training=True):
        for layer in self.layers:
            if isinstance(layer, Dropout):
                layer.training = training
    
    def forward(self, x):
        # Выравниваем вход сразу
        if len(x.shape) == 3:
            x = x.reshape(x.shape[0], -1)
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def backward(self, doutput):
        for layer in reversed(self.layers):
            if isinstance(layer, SimpleDenseLayer):
                doutput = layer.backward(doutput, self.learning_rate)
            elif isinstance(layer, (ReLU, Softmax, Dropout)):
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

def load_tiny_mnist():
    """Загрузка очень маленького набора данных"""
    try:
        import tensorflow as tf
        print("Загрузка tiny-MNIST...")
        (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
        
        # ОЧЕНЬ маленькая часть данных
        train_samples = 500   # Всего 500 примеров!
        test_samples = 100
        
        X_train = X_train[:train_samples] / 255.0
        y_train = y_train[:train_samples]
        X_test = X_test[:test_samples] / 255.0
        y_test = y_test[:test_samples]
        
        # One-hot encoding
        y_train_onehot = np.eye(10)[y_train]
        y_test_onehot = np.eye(10)[y_test]
        
        print(f"Загружено {X_train.shape[0]} тренировочных и {X_test.shape[0]} тестовых изображений")
        return X_train, y_train_onehot, X_test, y_test_onehot
        
    except ImportError:
        print("Установка TensorFlow...")
        !pip install tensorflow -q
        import tensorflow as tf
        return load_tiny_mnist()

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

def ultra_fast_training():
    print("⚡ СВЕРХБЫСТРОЕ ОБУЧЕНИЕ CNN")
    print("="*45)
    
    X_train_full, y_train_full, X_test, y_test = load_tiny_mnist()
    X_train, y_train, X_val, y_val = split_validation_data(X_train_full, y_train_full, 0.2)
    
    print(f"Тренировочные: {X_train.shape[0]}, Валидационные: {X_val.shape[0]}, Тестовые: {X_test.shape[0]}")
    
    # Тестируем разные dropout rates
    dropout_rates = [0.0, 0.2, 0.4, 0.6]
    results = {}
    training_histories = {}
    
    for dropout_rate in dropout_rates:
        print(f"\n--- Dropout Rate: {dropout_rate} ---")
        cnn = SimpleCNN(learning_rate=0.02, dropout_rate=dropout_rate)  # Увеличили learning rate
        
        epochs = 3  # Всего 3 эпохи!
        batch_size = 16  # Очень маленький batch size
        
        train_acc_history = []
        val_acc_history = []
        train_loss_history = []
        
        for epoch in range(epochs):
            # Очень быстрый training loop
            permutation = np.random.permutation(X_train.shape[0])
            X_shuffled = X_train[permutation]
            y_shuffled = y_train[permutation]
            
            epoch_loss = 0
            batch_count = 0
            
            # Только 5 батчей для СУПЕР скорости!
            for i in range(0, min(X_train.shape[0], 80), batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                y_pred = cnn.forward(X_batch)
                loss = cnn.compute_loss(y_pred, y_batch)
                epoch_loss += loss
                batch_count += 1
                doutput = y_pred - y_batch
                cnn.backward(doutput)
            
            avg_loss = epoch_loss / batch_count
            train_acc = cnn.accuracy(X_train[:100], y_train[:100])  # Только 100 примеров для оценки
            val_acc = cnn.accuracy(X_val[:50], y_val[:50])
            
            train_acc_history.append(train_acc)
            val_acc_history.append(val_acc)
            train_loss_history.append(avg_loss)
            
            print(f"Эпоха {epoch+1}: Loss = {avg_loss:.3f}, Train Acc = {train_acc:.3f}, Val Acc = {val_acc:.3f}")
        
        test_acc = cnn.accuracy(X_test[:50], y_test[:50])
        results[dropout_rate] = {
            'train_acc': train_acc_history[-1],
            'val_acc': val_acc_history[-1],
            'test_acc': test_acc,
            'final_loss': train_loss_history[-1]
        }
        training_histories[dropout_rate] = {
            'train_acc': train_acc_history,
            'val_acc': val_acc_history,
            'train_loss': train_loss_history
        }
        print(f"Финальная Test Accuracy: {test_acc:.3f}")
    
    return results, training_histories

def create_comprehensive_plots(results, training_histories):
    print("\n📊 КОМПЛЕКСНАЯ ВИЗУАЛИЗАЦИЯ РЕЗУЛЬТАТОВ")
    
    # Создаем большую фигуру с множеством графиков
    fig = plt.figure(figsize=(20, 15))
    
    # График 1: Сравнение точности по эпохам для разных dropout rates
    ax1 = plt.subplot(3, 4, 1)
    colors = ['red', 'blue', 'green', 'orange']
    for i, dropout_rate in enumerate(training_histories.keys()):
        history = training_histories[dropout_rate]
        ax1.plot(history['train_acc'], label=f'Dropout {dropout_rate}', 
                color=colors[i], marker='o', linewidth=2)
    
    ax1.set_title('Train Accuracy по эпохам', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1)
    
    # График 2: Validation accuracy по эпохам
    ax2 = plt.subplot(3, 4, 2)
    for i, dropout_rate in enumerate(training_histories.keys()):
        history = training_histories[dropout_rate]
        ax2.plot(history['val_acc'], label=f'Dropout {dropout_rate}', 
                color=colors[i], marker='s', linewidth=2)
    
    ax2.set_title('Validation Accuracy по эпохам', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)
    
    # График 3: Финальная test accuracy
    ax3 = plt.subplot(3, 4, 3)
    dropout_rates = list(results.keys())
    test_accs = [results[dr]['test_acc'] for dr in dropout_rates]
    
    bars = ax3.bar(dropout_rates, test_accs, alpha=0.7, color=colors[:len(dropout_rates)])
    ax3.set_title('Финальная Test Accuracy', fontsize=12, fontweight='bold')
    ax3.set_xlabel('Dropout Rate')
    ax3.set_ylabel('Accuracy')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(0, 1)
    
    # Добавляем значения на столбцы
    for bar, acc in zip(bars, test_accs):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # График 4: Сравнение train vs val accuracy для всех моделей
    ax4 = plt.subplot(3, 4, 4)
    final_train_accs = [results[dr]['train_acc'] for dr in dropout_rates]
    final_val_accs = [results[dr]['val_acc'] for dr in dropout_rates]
    
    x_pos = np.arange(len(dropout_rates))
    width = 0.35
    
    ax4.bar(x_pos - width/2, final_train_accs, width, label='Train Acc', alpha=0.7)
    ax4.bar(x_pos + width/2, final_val_accs, width, label='Val Acc', alpha=0.7)
    
    ax4.set_title('Финальная Train vs Validation Accuracy', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Dropout Rate')
    ax4.set_ylabel('Accuracy')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(dropout_rates)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1)
    
    # График 5: Функции потерь по эпохам
    ax5 = plt.subplot(3, 4, 5)
    for i, dropout_rate in enumerate(training_histories.keys()):
        history = training_histories[dropout_rate]
        ax5.plot(history['train_loss'], label=f'Dropout {dropout_rate}', 
                color=colors[i], marker='d', linewidth=2)
    
    ax5.set_title('Функция потерь по эпохам', fontsize=12, fontweight='bold')
    ax5.set_xlabel('Эпоха')
    ax5.set_ylabel('Loss')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # График 6: Разница между train и val accuracy (overfitting)
    ax6 = plt.subplot(3, 4, 6)
    overfitting_gap = [final_train_accs[i] - final_val_accs[i] for i in range(len(dropout_rates))]
    
    bars = ax6.bar(dropout_rates, overfitting_gap, alpha=0.7, 
                  color=['red' if gap > 0.1 else 'green' for gap in overfitting_gap])
    ax6.set_title('Разница Train-Val (Overfitting)', fontsize=12, fontweight='bold')
    ax6.set_xlabel('Dropout Rate')
    ax6.set_ylabel('Train Acc - Val Acc')
    ax6.grid(True, alpha=0.3)
    
    # График 7: Производительность по dropout rates (радарная диаграмма)
    ax7 = plt.subplot(3, 4, 7, polar=True)
    metrics = ['Train Acc', 'Val Acc', 'Test Acc', 'Generalization']
    num_vars = len(metrics)
    
    # Вычисляем значения для радарной диаграммы
    values = {}
    for dr in dropout_rates:
        values[dr] = [
            results[dr]['train_acc'],
            results[dr]['val_acc'], 
            results[dr]['test_acc'],
            min(results[dr]['val_acc'], results[dr]['test_acc'])  # Обобщающая способность
        ]
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # Замыкаем круг
    
    for i, dr in enumerate(dropout_rates):
        vals = values[dr]
        vals += vals[:1]  # Замыкаем круг
        ax7.plot(angles, vals, 'o-', linewidth=2, label=f'Dropout {dr}', color=colors[i])
        ax7.fill(angles, vals, alpha=0.1, color=colors[i])
    
    ax7.set_yticklabels([])
    ax7.set_xticks(angles[:-1])
    ax7.set_xticklabels(metrics)
    ax7.set_title('Сравнение моделей\n(Радарная диаграмма)', fontsize=12, fontweight='bold')
    ax7.legend(bbox_to_anchor=(1.1, 1.1))
    
    # График 8: Heatmap эффективности
    ax8 = plt.subplot(3, 4, 8)
    performance_matrix = np.array([
        [results[dr]['train_acc'] for dr in dropout_rates],
        [results[dr]['val_acc'] for dr in dropout_rates],
        [results[dr]['test_acc'] for dr in dropout_rates]
    ])
    
    im = ax8.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax8.set_xticks(range(len(dropout_rates)))
    ax8.set_xticklabels(dropout_rates)
    ax8.set_yticks(range(3))
    ax8.set_yticklabels(['Train', 'Val', 'Test'])
    ax8.set_title('Матрица эффективности', fontsize=12, fontweight='bold')
    
    # Добавляем значения в heatmap
    for i in range(3):
        for j in range(len(dropout_rates)):
            text = ax8.text(j, i, f'{performance_matrix[i, j]:.2f}',
                           ha="center", va="center", color="black", fontweight='bold')
    
    # График 9: Эволюция overfitting по эпохам
    ax9 = plt.subplot(3, 4, 9)
    for i, dropout_rate in enumerate(training_histories.keys()):
        history = training_histories[dropout_rate]
        overfitting_epochs = [history['train_acc'][j] - history['val_acc'][j] 
                            for j in range(len(history['train_acc']))]
        ax9.plot(overfitting_epochs, label=f'Dropout {dropout_rate}', 
                color=colors[i], marker='^', linewidth=2)
    
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax9.set_title('Эволюция Overfitting', fontsize=12, fontweight='bold')
    ax9.set_xlabel('Эпоха')
    ax9.set_ylabel('Train Acc - Val Acc')
    ax9.legend()
    ax9.grid(True, alpha=0.3)
    
    # График 10: Сравнение всех метрик для лучшей модели
    ax10 = plt.subplot(3, 4, 10)
    best_dropout = max(results.keys(), key=lambda x: results[x]['test_acc'])
    best_results = results[best_dropout]
    
    metrics_names = ['Train Acc', 'Val Acc', 'Test Acc', 'Loss']
    metrics_values = [
        best_results['train_acc'],
        best_results['val_acc'],
        best_results['test_acc'],
        1 - best_results['final_loss']  # Инвертируем loss для визуализации
    ]
    
    bars = ax10.bar(metrics_names, metrics_values, alpha=0.7, color=['blue', 'green', 'red', 'purple'])
    ax10.set_title(f'Лучшая модель\n(Dropout={best_dropout})', fontsize=12, fontweight='bold')
    ax10.set_ylabel('Значение')
    ax10.grid(True, alpha=0.3)
    ax10.set_ylim(0, 1)
    
    for bar, value in zip(bars, metrics_values):
        height = bar.get_height()
        ax10.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                 f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # График 11: Скорость сходимости
    ax11 = plt.subplot(3, 4, 11)
    for i, dropout_rate in enumerate(training_histories.keys()):
        history = training_histories[dropout_rate]
        ax11.plot(history['train_acc'], label=f'Dropout {dropout_rate}', 
                 color=colors[i], linewidth=2)
        # Отмечаем финальную точку
        ax11.scatter(len(history['train_acc'])-1, history['train_acc'][-1], 
                    color=colors[i], s=100, zorder=5)
    
    ax11.set_title('Скорость сходимости', fontsize=12, fontweight='bold')
    ax11.set_xlabel('Эпоха')
    ax11.set_ylabel('Train Accuracy')
    ax11.legend()
    ax11.grid(True, alpha=0.3)
    ax11.set_ylim(0, 1)
    
    # График 12: Сводная таблица результатов
    ax12 = plt.subplot(3, 4, 12)
    ax12.axis('off')
    
    # Создаем текстовую таблицу
    table_data = []
    headers = ['Dropout', 'Train Acc', 'Val Acc', 'Test Acc', 'Loss']
    table_data.append(headers)
    
    for dr in dropout_rates:
        row = [
            f'{dr}',
            f'{results[dr]["train_acc"]:.3f}',
            f'{results[dr]["val_acc"]:.3f}',
            f'{results[dr]["test_acc"]:.3f}',
            f'{results[dr]["final_loss"]:.3f}'
        ]
        table_data.append(row)
    
    table = ax12.table(cellText=table_data, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax12.set_title('Сводная таблица результатов', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('comprehensive_results.png', dpi=120, bbox_inches='tight')
    plt.show()
    
    return best_dropout

def main():
    print("⚡ ЗАПУСК СВЕРХБЫСТРОГО ОБУЧЕНИЯ")
    print("Используется ОЧЕНЬ маленький набор данных: 500 примеров")
    print("Ожидаемое время выполнения: 30-60 секунд ⚡")
    print("="*55)
    
    # Сверхбыстрое обучение
    results, training_histories = ultra_fast_training()
    
    # Комплексная визуализация
    best_dropout = create_comprehensive_plots(results, training_histories)
    
    # Вывод итогов
    print("\n" + "="*60)
    print("🎯 ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
    print("="*60)
    for dropout_rate in sorted(results.keys()):
        res = results[dropout_rate]
        print(f"Dropout {dropout_rate}: Train={res['train_acc']:.3f}, "
              f"Val={res['val_acc']:.3f}, Test={res['test_acc']:.3f}, Loss={res['final_loss']:.3f}")
    
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: Dropout {best_dropout}")
    print(f"   Test Accuracy: {results[best_dropout]['test_acc']:.3f}")
    
    # Скачиваем результаты
    print("\n📥 Скачивание комплексного графика...")
    files.download('comprehensive_results.png')
    
    print("\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО! (Время выполнения: ~30 секунд)")

if __name__ == "__main__":
    main()
