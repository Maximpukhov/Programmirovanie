#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>

int main() {
    int server_socket;
    struct sockaddr_in server_addr, client_addr;
    socklen_t client_len = sizeof(client_addr);
    char buffer[1024];

    // Создание UDP сокета
    server_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (server_socket < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0; // Порт выбирается автоматически

    // Привязка сокета
    if (bind(server_socket, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("Bind failed");
        exit(EXIT_FAILURE);
    }

    // Получение и вывод выбранного порта
    socklen_t len = sizeof(server_addr);
    getsockname(server_socket, (struct sockaddr *)&server_addr, &len);
    std::cout << "Server is running on port: " << ntohs(server_addr.sin_port) << std::endl;

    while (true) {
        // Получение данных от клиента
        int n = recvfrom(server_socket, buffer, sizeof(buffer), 0,
                         (struct sockaddr *)&client_addr, &client_len);
        if (n < 0) {
            perror("recvfrom failed");
            continue;
        }
        buffer[n] = '\0';

        // Вывод информации о клиенте
        std::cout << "Received from client: " << buffer
                  << " | IP: " << inet_ntoa(client_addr.sin_addr)
                  << " | Port: " << ntohs(client_addr.sin_port) << std::endl;

        // Преобразование данных (число + 1)
        int received_number = atoi(buffer);
        received_number++;
        std::string response = std::to_string(received_number);

        // Отправка ответа клиенту
        sendto(server_socket, response.c_str(), response.size(), 0,
               (struct sockaddr *)&client_addr, client_len);
    }

    close(server_socket);
    return 0;
}
