#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cout << "Использование: " << argv[0] << " <IP адрес> <порт>\n";
        return 1;
    }

    // Создаем TCP сокет
    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket < 0) {
        std::cerr << "Ошибка создания сокета\n";
        return 1;
    }

    // Настраиваем адрес сервера
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(std::stoi(argv[2]));
    if (inet_aton(argv[1], &server_addr.sin_addr) == 0) {
        std::cerr << "Неверный IP адрес\n";
        return 1;
    }

    // Подключаемся к серверу
    if (connect(client_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Ошибка подключения\n";
        return 1;
    }

    // Запрашиваем число и задержку у пользователя
    int number;
    std::cout << "Введите число (1-10): ";
    std::cin >> number;

    if (number < 1 || number > 10) {
        std::cerr << "Число должно быть от 1 до 10\n";
        close(client_socket);
        return 1;
    }

    // Отправляем число в цикле
    for(int i = 0; i < 10; i++) { // 10 итераций для примера
        std::string message = std::to_string(number);
        send(client_socket, message.c_str(), message.length(), 0);
        
        std::cout << "Отправлено число " << number 
                  << ", ожидание " << number << " секунд...\n";
        sleep(number);  // Задержка равна отправляемому числу
    }

    close(client_socket);
    return 0;
}
