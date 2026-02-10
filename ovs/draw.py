import matplotlib.pyplot as plt
import numpy as np
import matplotlib

# Настройки для русского языка
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# Данные из условия
net_time = 64174  # время для сети (InfiniBand/Gigabit Ethernet)
qpi_time = 64158  # время для QPI
sm_time = 64140   # время для оперативной памяти (SMP/NUMA)

# Создаем диапазон размеров сообщений (от 1KB до 100MB)
sizes_kb = np.logspace(0, 5, 50)  # от 1KB до 100MB
sizes_bytes = sizes_kb * 1024

# Предполагаем, что данные даны для 1MB сообщения
base_size = 1024 * 1024  # 1MB

# Рассчитываем коэффициенты пропускной способности
k_net = net_time / base_size
k_qpi = qpi_time / base_size
k_sm = sm_time / base_size

# Рассчитываем время передачи
t_net = k_net * sizes_bytes
t_qpi = k_qpi * sizes_bytes
t_sm = k_sm * sizes_bytes

# Создаем основной график
plt.figure(figsize=(14, 8))

# Основной график
plt.subplot(1, 2, 1)
plt.loglog(sizes_kb, t_net, 'b-', linewidth=3, label=f'Сеть (InfiniBand/GE) - {net_time} мкс')
plt.loglog(sizes_kb, t_qpi, 'r--', linewidth=3, label=f'QPI - {qpi_time} мкс')
plt.loglog(sizes_kb, t_sm, 'g:', linewidth=3, label=f'Оперативная память - {sm_time} мкс')

plt.xlabel('Размер сообщения, KB (логарифмическая шкала)', fontsize=12)
plt.ylabel('Время передачи, микросекунды (логарифмическая шкала)', fontsize=12)
plt.title('Зависимость времени передачи от размера сообщения\nдля разных уровней коммуникационной среды', 
          fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.legend(loc='upper left', fontsize=10)
plt.axvline(x=1024, color='gray', linestyle=':', alpha=0.5)  # линия для 1MB

# Добавляем аннотации
plt.annotate('1 MB', xy=(1024, 100), xytext=(1024, 1000),
             arrowprops=dict(arrowstyle='->', color='gray'),
             fontsize=10, color='gray')

# График пропускной способности
plt.subplot(1, 2, 2)

# Рассчитываем пропускную способность в MB/сек
bandwidth_net = (sizes_bytes / (1024 * 1024)) / (t_net / 1e6)  # MB/сек
bandwidth_qpi = (sizes_bytes / (1024 * 1024)) / (t_qpi / 1e6)  # MB/сек
bandwidth_sm = (sizes_bytes / (1024 * 1024)) / (t_sm / 1e6)    # MB/сек

plt.semilogx(sizes_kb, bandwidth_net, 'b-', linewidth=3, label='Сеть (InfiniBand/GE)')
plt.semilogx(sizes_kb, bandwidth_qpi, 'r--', linewidth=3, label='QPI')
plt.semilogx(sizes_kb, bandwidth_sm, 'g:', linewidth=3, label='Оперативная память')

plt.xlabel('Размер сообщения, KB (логарифмическая шкала)', fontsize=12)
plt.ylabel('Пропускная способность, MB/сек', fontsize=12)
plt.title('Эффективная пропускная способность', fontsize=14, fontweight='bold', pad=20)
plt.grid(True, alpha=0.3, which='both', linestyle='--')
plt.legend(loc='lower right', fontsize=10)

plt.tight_layout()
plt.savefig('communication_analysis.png', dpi=300, bbox_inches='tight')
print("График сохранен как 'communication_analysis.png'")

# Вывод дополнительной информации
print("\nАнализ данных:")
print("=" * 50)
print(f"1. Для сообщения 1MB:")
print(f"   - Сеть: {net_time} мкс")
print(f"   - QPI: {qpi_time} мкс")
print(f"   - Оперативная память: {sm_time} мкс")
print(f"\n2. Относительная производительность (чем меньше, тем лучше):")
print(f"   - QPI медленнее сети в {qpi_time/net_time:.2f} раз")
print(f"   - Оперативная память медленнее сети в {sm_time/net_time:.2f} раз")
print(f"\n3. Расчетная пропускная способность для 1MB:")
print(f"   - Сеть: {1/(net_time/1e6):.2f} MB/сек")
print(f"   - QPI: {1/(qpi_time/1e6):.2f} MB/сек")
print(f"   - Оперативная память: {1/(sm_time/1e6):.2f} MB/сек")

plt.show()
