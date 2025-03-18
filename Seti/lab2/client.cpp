#include <iostream>
#include <cstring>
#include <unistd.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc != 3) {
        cerr << "Usage: " << argv[0] << " <server_ip> <server_port>" << endl;
        return 1;
    }

    int client_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (client_socket == -1) {
        perror("Socket creation failed");
        return 1;
    }

    sockaddr_in server_addr{};
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(atoi(argv[2]));
    inet_pton(AF_INET, argv[1], &server_addr.sin_addr);

    if (connect(client_socket, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
        perror("Connection failed");
        close(client_socket);
        return 1;
    }

    cout << "Connected to server." << endl;

    int i;
    cout << "Enter a number (i): ";
    cin >> i;

    for (int count = 0; count < i; ++count) {
        string message = "Message " + to_string(count + 1) + " from client";
        send(client_socket, message.c_str(), message.size(), 0);
        sleep(i);
    }

    close(client_socket);
    cout << "Disconnected from server." << endl;
    return 0;
}
