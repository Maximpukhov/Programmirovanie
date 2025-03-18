#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/wait.h>
#include <signal.h>

using namespace std;

const int BUFFER_SIZE = 1024;

void handle_client(int client_socket, sockaddr_in client_addr) {
    char buffer[BUFFER_SIZE];
    char client_ip[INET_ADDRSTRLEN];
    inet_ntop(AF_INET, &(client_addr.sin_addr), client_ip, INET_ADDRSTRLEN);
    cout << "Client connected: " << client_ip << ":" << ntohs(client_addr.sin_port) << endl;

    while (true) {
        ssize_t bytes_received = recv(client_socket, buffer, BUFFER_SIZE, 0);
        if (bytes_received <= 0) {
            break;
        }
        buffer[bytes_received] = '\0';
        cout << "Received from " << client_ip << ": " << buffer << endl;
    }

    close(client_socket);
    cout << "Client disconnected: " << client_ip << endl;
    exit(0);
}

void sigchld_handler(int sig) {
    while (waitpid(-1, nullptr, WNOHANG) > 0);
}

int main() {
    int server_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (server_socket == -1) {
        perror("Socket creation failed");
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = INADDR_ANY;
    server_addr.sin_port = 0;

    if (bind(server_socket, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("Bind failed");
        close(server_socket);
        return 1;
    }

    socklen_t addr_len = sizeof(server_addr);
    getsockname(server_socket, (sockaddr*)&server_addr, &addr_len);
    cout << "Server is running on port: " << ntohs(server_addr.sin_port) << endl;

    if (listen(server_socket, 5) == -1) {
        perror("Listen failed");
        close(server_socket);
        return 1;
    }

    signal(SIGCHLD, sigchld_handler);

    while (true) {
        sockaddr_in client_addr{};
        socklen_t client_len = sizeof(client_addr);
        int client_socket = accept(server_socket, (sockaddr*)&client_addr, &client_len);
        if (client_socket == -1) {
            perror("Accept failed");
            continue;
        }

        pid_t pid = fork();
        if (pid == 0) {
            close(server_socket);
            handle_client(client_socket, client_addr);
        } else if (pid > 0) {
            close(client_socket);
        } else {
            perror("Fork failed");
        }
    }

    close(server_socket);
    return 0;
}
