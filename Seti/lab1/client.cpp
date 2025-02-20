#include <iostream>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <server_ip> <server_port> <i> <number_of_loops>" << std::endl;
        return EXIT_FAILURE;
    }

    const char *server_ip = argv[1];
    int server_port = atoi(argv[2]);
    int i = atoi(argv[3]);
    int loops = atoi(argv[4]);

    int client_socket;
    struct sockaddr_in server_addr;
    char buffer[1024];

    // Создание UDP сокета
    client_socket = socket(AF_INET, SOCK_DGRAM, 0);
    if (client_socket < 0) {
        perror("Socket creation failed");
        exit(EXIT_FAILURE);
    }

    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(server_port);
    if (inet_pton(AF_INET, server_ip, &server_addr.sin_addr) <= 0) {
        perror("Invalid address");
        exit(EXIT_FAILURE);
    }

    // Циклическая отправка данных
    for (int count = 0; count < loops; count++) {
        std::string message = std::to_string(i);
        sendto(client_socket, message.c_str(), message.size(), 0,
               (struct sockaddr *)&server_addr, sizeof(server_addr));
        std::cout << "Sent: " << i << std::endl;

        // Получение ответа от сервера
        socklen_t len = sizeof(server_addr);
        int n = recvfrom(client_socket, buffer, sizeof(buffer), 0,
                         (struct sockaddr *)&server_addr, &len);
        if (n < 0) {
            perror("recvfrom failed");
            continue;
        }
        buffer[n] = '\0';
        std::cout << "Received from server: " << buffer << std::endl;

        sleep(i); // Задержка в i секунд
    }

    close(client_socket);
    return 0;
}
