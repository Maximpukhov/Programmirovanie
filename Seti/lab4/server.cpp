// server.cpp
#include <iostream>
#include <string>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <fcntl.h>
#include <set>

int main() {
    // Создаем TCP сокет
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket < 0) {
        std::cerr << "Ошибка создания сокета\n";
        return 1;
    }

    // Для повторного использования адреса
    int opt = 1;
    setsockopt(server_socket, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

    // Настраиваем адрес сервера
    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0; // Автоматический выбор порта

    // Привязываем сокет
    if (bind(server_socket, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
        std::cerr << "Ошибка привязки сокета\n";
        return 1;
    }

    // Получаем информацию о выделенном порте
    socklen_t len = sizeof(server_addr);
    if (getsockname(server_socket, (struct sockaddr*)&server_addr, &len) < 0) {
        std::cerr << "Ошибка получения информации о сокете\n";
        return 1;
    }

    std::cout << "Сервер запущен на порту: " << ntohs(server_addr.sin_port) << std::endl;

    // Слушаем входящие подключения
    listen(server_socket, 5);

    // Множество для хранения активных сокетов
    std::set<int> active_sockets;
    active_sockets.insert(server_socket);

    while (true) {
        // Подготовка множеств файловых дескрипторов для select
        fd_set read_fds;
        FD_ZERO(&read_fds);
        
        // Находим максимальный дескриптор
        int max_fd = server_socket;
        for (int socket : active_sockets) {
            FD_SET(socket, &read_fds);
            if (socket > max_fd) max_fd = socket;
        }

        // Ожидаем события на сокетах
        if (select(max_fd + 1, &read_fds, NULL, NULL, NULL) < 0) {
            std::cerr << "Ошибка в select()\n";
            break;
        }

        // Проверяем все сокеты
        std::set<int> sockets_to_remove;
        for (int socket : active_sockets) {
            if (FD_ISSET(socket, &read_fds)) {
                if (socket == server_socket) {
                    // Новое подключение
                    sockaddr_in client_addr{};
                    socklen_t client_len = sizeof(client_addr);
                    int client_socket = accept(server_socket, 
                                            (struct sockaddr*)&client_addr, 
                                            &client_len);
                    
                    if (client_socket < 0) {
                        std::cerr << "Ошибка принятия подключения\n";
                        continue;
                    }

                    std::cout << "Новое подключение от " 
                              << inet_ntoa(client_addr.sin_addr) 
                              << ":" << ntohs(client_addr.sin_port) << std::endl;

                    active_sockets.insert(client_socket);
                }
                else {
                    // Чтение данных от клиента
                    char buffer[1024];
                    int recv_size = recv(socket, buffer, sizeof(buffer), 0);

                    if (recv_size <= 0) {
                        // Клиент отключился
                        sockets_to_remove.insert(socket);
                    }
                    else {
                        buffer[recv_size] = '\0';
                        std::cout << "Получено: " << buffer << std::endl;
                    }
                }
            }
        }

        // Удаляем закрытые сокеты
        for (int socket : sockets_to_remove) {
            close(socket);
            active_sockets.erase(socket);
        }
    }

    // Закрываем все сокеты
    for (int socket : active_sockets) {
        close(socket);
    }

    return 0;
}
