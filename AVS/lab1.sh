echo "=== Информация о системе ==="
echo "Название ОС: $(lsb_release -d | cut -f2)"
echo "Версия ОС: $(lsb_release -r | cut -f2)"
echo "Версия ядра: $(uname -r)"
echo "Архитектура: $(uname -m)"
echo

echo "=== Информация о процессоре ==="
echo "Модель процессора: $(lscpu | grep 'Model name' | awk -F: '{print $2}' | xargs)"
echo "Частота процессора: $(lscpu | grep 'CPU MHz' | awk -F: '{print $2}' | xargs)"
echo "Количество ядер: $(lscpu | grep 'Thread(s) per core:' | awk -F: '{print $2}'| xargs)"
echo "Размер кэш-памяти: $(lscpu | grep 'L1d cache' | awk -F: '{print $2}' | xargs)"
echo "Количество доступных процессоров: $(lscpu | grep 'Socket(s):' | awk '{print $2}')"
echo

echo "=== Информация о оперативной памяти ==="
read total used free <<< $(free -h | awk '/Mem:/ {print $2, $3, $7}')
echo "Общий размер: $total"
echo "Использованный размер: $used"
echo "Доступный размер: $free"
echo

echo "=== Информация о сетевых интерфейсах ==="
for iface in $(ls /sys/class/net/); do
    echo "Интерфейс: $iface"
    ip addr show $iface | grep -E 'inet | link/ether' | awk '{print $1, $2}'
    ethtool $iface | grep 'Speed'
    echo
done

echo "=== Информация о системных разделах ==="
df -h --output=source,target,size,used,avail
echo

echo "=== Скрипт выполнен успешно ==="
