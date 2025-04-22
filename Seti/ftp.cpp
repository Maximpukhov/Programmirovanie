#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <fstream>
#include <stdexcept>
#include <cstring>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string_view>

class FTPClient {
public:
    FTPClient(const std::string& host, int port, const std::string& username, const std::string& password)
        : host_(host), port_(port), username_(username), password_(password), control_sock_(-1) {}

    ~FTPClient() {
        if (control_sock_ != -1) close(control_sock_);
    }

    void connectToServer() {
        control_sock_ = socket(AF_INET, SOCK_STREAM, 0);
        if (control_sock_ == -1) {
            throw std::runtime_error("Не удалось создать сокет: " + std::string(strerror(errno)));
        }

        sockaddr_in server_addr{};
        server_addr.sin_family = AF_INET;
        server_addr.sin_port = htons(port_);
        if (inet_pton(AF_INET, host_.c_str(), &server_addr.sin_addr) <= 0) {
            throw std::runtime_error("Неверный адрес: " + host_ + " (используйте '127.0.0.1' для локального хоста)");
        }

        if (connect(control_sock_, (sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
            throw std::runtime_error("Не удалось подключиться к FTP-серверу по адресу " + host_ + ":" + 
                                    std::to_string(port_) + ": " + std::string(strerror(errno)));
        }

        std::string response = readResponse();
        if (!response.starts_with("220")) {
            throw std::runtime_error("Сервер не принял соединение: " + response);
        }
        std::cout << "Подключено к FTP-серверу: " << host_ << ":" << port_ << std::endl;
    }

    void login() {
        sendCommand("USER " + username_);
        if (!readResponse().starts_with("331")) {
            throw std::runtime_error("Имя пользователя не принято");
        }

        sendCommand("PASS " + password_);
        if (!readResponse().starts_with("230")) {
            throw std::runtime_error("Пароль не принят");
        }
        std::cout << "Вход выполнен как " << username_ << std::endl;
    }

    bool enterPassiveMode(std::string& ip, int& port) {
        sendCommand("PASV");
        std::string response = readResponse();
        if (!response.starts_with("227")) {
            std::cerr << "Пассивный режим не поддерживается: " << response << std::endl;
            return false;
        }

        size_t start = response.find('(');
        size_t end = response.find(')');
        if (start == std::string::npos || end == std::string::npos) {
            std::cerr << "Неверный формат ответа PASV" << std::endl;
            return false;
        }

        std::vector<int> values;
        std::stringstream ss(response.substr(start + 1, end - start - 1));
        std::string token;
        while (std::getline(ss, token, ',')) {
            values.push_back(std::stoi(token));
        }

        if (values.size() != 6) {
            std::cerr << "Неверные данные PASV" << std::endl;
            return false;
        }
        ip = std::to_string(values[0]) + "." + std::to_string(values[1]) + "." +
            std::to_string(values[2]) + "." + std::to_string(values[3]);
        port = values[4] * 256 + values[5];
        return true;
    }

    void downloadFile(const std::string& remotePath, const std::string& localPath) {
        sendCommand("TYPE I");
        if (!readResponse().starts_with("200")) {
            throw std::runtime_error("Не удалось установить бинарный режим");
        }

        std::string data_ip;
        int data_port;
        if (!enterPassiveMode(data_ip, data_port)) {
            throw std::runtime_error("Не удалось войти в пассивный режим");
        }

        int data_sock = createDataSocket(data_ip, data_port);
        sendCommand("RETR " + remotePath);
        if (!readResponse().starts_with("150")) {
            close(data_sock);
            throw std::runtime_error("Сервер отклонил команду RETR");
        }

        std::ofstream file(localPath, std::ios::binary);
        if (!file) {
            close(data_sock);
            throw std::runtime_error("Не удалось открыть локальный файл: " + localPath);
        }

        char buffer[4096];
        ssize_t bytes_received;
        while ((bytes_received = recv(data_sock, buffer, sizeof(buffer), 0)) > 0) {
            file.write(buffer, bytes_received);
        }
        if (bytes_received < 0) {
            throw std::runtime_error("Ошибка получения данных: " + std::string(strerror(errno)));
        }
        file.close();
        close(data_sock);

        if (!readResponse().starts_with("226")) {
            throw std::runtime_error("Передача файла не удалась");
        }
        std::cout << "Файл загружен: " << localPath << std::endl;
    }

    void uploadFile(const std::string& localPath, const std::string& remotePath) {
        sendCommand("TYPE I");
        if (!readResponse().starts_with("200")) {
            throw std::runtime_error("Не удалось установить бинарный режим");
        }

        std::ifstream file(localPath, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Не удалось открыть локальный файл: " + localPath);
        }

        std::string data_ip;
        int data_port;
        if (!enterPassiveMode(data_ip, data_port)) {
            throw std::runtime_error("Не удалось войти в пассивный режим");
        }

        int data_sock = createDataSocket(data_ip, data_port);
        sendCommand("STOR " + remotePath);
        if (!readResponse().starts_with("150")) {
            close(data_sock);
            throw std::runtime_error("Сервер отклонил команду STOR");
        }

        char buffer[4096];
        while (file.read(buffer, sizeof(buffer)) || file.gcount() > 0) {
            if (send(data_sock, buffer, file.gcount(), 0) < 0) {
                throw std::runtime_error("Ошибка отправки данных: " + std::string(strerror(errno)));
            }
        }
        file.close();
        close(data_sock);

        if (!readResponse().starts_with("226")) {
            throw std::runtime_error("Передача файла не удалась");
        }
        std::cout << "Файл загружен на сервер: " << remotePath << std::endl;
    }

    void listFiles() {
        sendCommand("PASV");
        std::string response = readResponse();
        if (!response.starts_with("227")) {
            throw std::runtime_error("Не удалось войти в пассивный режим");
        }

        std::string data_ip;
        int data_port;
        if (!enterPassiveMode(data_ip, data_port)) {
            throw std::runtime_error("Не удалось разобрать ответ PASV");
        }

        int data_sock = createDataSocket(data_ip, data_port);
        sendCommand("LIST");
        if (!readResponse().starts_with("150")) {
            close(data_sock);
            throw std::runtime_error("Сервер отклонил команду LIST");
        }

        char buffer[4096];
        ssize_t bytes_received;
        std::string file_list;
        while ((bytes_received = recv(data_sock, buffer, sizeof(buffer), 0)) > 0) {
            file_list.append(buffer, bytes_received);
        }
        close(data_sock);

        if (!readResponse().starts_with("226")) {
            throw std::runtime_error("Получение списка файлов не удалось");
        }

        std::cout << "Список файлов:\n" << file_list << std::endl;
    }

    void createDirectory(const std::string& directory) {
        sendCommand("MKD " + directory);
        std::string response = readResponse();
        if (!response.starts_with("257")) {
            throw std::runtime_error("Не удалось создать папку: " + response);
        }
        std::cout << "Папка создана: " << directory << std::endl;
    }

    void changeDirectory(const std::string& directory) {
        sendCommand("CWD " + directory);
        std::string response = readResponse();
        if (!response.starts_with("250")) {
            throw std::runtime_error("Не удалось сменить каталог: " + response);
        }
        std::cout << "Текущий каталог изменен на: " << directory << std::endl;
    }

    std::string getCurrentDirectory() {
        sendCommand("PWD");
        std::string response = readResponse();
        if (!response.starts_with("257")) {
            throw std::runtime_error("Не удалось получить текущий каталог: " + response);
        }
        size_t start = response.find('"');
        size_t end = response.rfind('"');
        if (start != std::string::npos && end != std::string::npos && start != end) {
            return response.substr(start + 1, end - start - 1);
        }
        return response;
    }
    
    void removeDirectory(const std::string& directory) {
        sendCommand("RMD " + directory);
        std::string response = readResponse();
        if (!response.starts_with("250")) {
            throw std::runtime_error("Не удалось удалить папку: " + response);
        }
        std::cout << "Папка удалена: " << directory << std::endl;
    }

    void deleteFile(const std::string& filePath) {
        sendCommand("DELE " + filePath);
        std::string response = readResponse();
        if (!response.starts_with("250")) {
            throw std::runtime_error("Не удалось удалить файл: " + response);
        }
        std::cout << "Файл удален: " << filePath << std::endl;
    }

    void renameFileOrDirectory(const std::string& oldName, const std::string& newName) {
        sendCommand("RNFR " + oldName);
        std::string response = readResponse();
        if (!response.starts_with("350")) {
            throw std::runtime_error("Не удалось выбрать файл/папку для переименования: " + response);
        }

        sendCommand("RNTO " + newName);
        response = readResponse();
        if (!response.starts_with("250")) {
            throw std::runtime_error("Не удалось переименовать: " + response);
        }

        std::cout << "Успешно переименовано: " << oldName << " -> " << newName << std::endl;
    }


private:
    int control_sock_;
    std::string host_;
    int port_;
    std::string username_;
    std::string password_;

    void sendCommand(std::string_view cmd) {
        std::string command = std::string(cmd) + "\r\n";
        if (send(control_sock_, command.c_str(), command.size(), 0) < 0) {
            throw std::runtime_error("Не удалось отправить команду: " + std::string(strerror(errno)));
        }
    }

    std::string readResponse() {
        std::string response;
        char buffer[1024];
        ssize_t bytes_received;
        while ((bytes_received = recv(control_sock_, buffer, sizeof(buffer) - 1, 0)) > 0) {
            buffer[bytes_received] = '\0';
            response += buffer;
            if (response.find("\r\n") != std::string::npos) break;
        }
        if (bytes_received < 0) {
            throw std::runtime_error("Ошибка чтения ответа: " + std::string(strerror(errno)));
        }
        return response;
    }

    int createDataSocket(const std::string& ip, int port) {
        int data_sock = socket(AF_INET, SOCK_STREAM, 0);
        if (data_sock == -1) {
            throw std::runtime_error("Не удалось создать сокет данных: " + std::string(strerror(errno)));
        }

        sockaddr_in data_addr{};
        data_addr.sin_family = AF_INET;
        data_addr.sin_port = htons(port);
        inet_pton(AF_INET, ip.c_str(), &data_addr.sin_addr);

        if (connect(data_sock, (sockaddr*)&data_addr, sizeof(data_addr)) == -1) {
            close(data_sock);
            throw std::runtime_error("Не удалось подключить сокет данных: " + std::string(strerror(errno)));
        }
        return data_sock;
    }
};

int main() {
    try {
        std::string server;
        int port = 21;
        std::string username;
        std::string password;
        
        std::cout << "Введите адрес сервера: ";
        std::cin >> server;
        std::cout << "Введите имя пользователя: ";
        std::cin >> username;
        std::cout << "Введите пароль: ";
        std::cin >> password;
        
        FTPClient ftp(server, port, username, password);
        ftp.connectToServer();
        ftp.login();
        
        while (true) {
            std::cout << "\nМеню:" << std::endl;
            std::cout << "1. Скачать файл" << std::endl;
            std::cout << "2. Загрузить файл" << std::endl;
            std::cout << "3. Показать список файлов" << std::endl;
            std::cout << "4. Создать каталог" << std::endl;
            std::cout << "5. Сменить каталог" << std::endl;
            std::cout << "6. Показать текущий каталог" << std::endl;
            std::cout << "7. Удалить каталог" << std::endl;
            std::cout << "8. Удалить файл" << std::endl;
            std::cout << "9. Переименовать файл или каталог" << std::endl;
            std::cout << "q. Выйти" << std::endl;
            std::cout << "Выберите действие: ";
            
            std::string choice;
            std::cin >> choice;
            
            if (choice == "1") {
                std::string remoteFile, localFile;
                std::cout << "Введите имя удаленного файла: ";
                std::cin >> remoteFile;
                std::cout << "Введите имя локального файла: ";
                std::cin >> localFile;
                try {
                    ftp.downloadFile(remoteFile, localFile);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "2") {
                std::string localFile, remoteFile;
                std::cout << "Введите имя локального файла: ";
                std::cin >> localFile;
                std::cout << "Введите имя удаленного файла: ";
                std::cin >> remoteFile;
                try{
                    ftp.uploadFile(localFile, remoteFile);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "3") {
                try {
                    ftp.listFiles();
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "4") {
                std::string directory;
                std::cout << "Введите имя каталога: ";
                std::cin >> directory;
                try {
                    ftp.createDirectory(directory);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "5") {
                std::string directory;
                std::cout << "Введите путь к каталогу: ";
                std::cin >> directory;
                try {
                    ftp.changeDirectory(directory);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "6") {
                try {
                    std::cout << "Текущий каталог: " << ftp.getCurrentDirectory() << std::endl;
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "7") {
                std::string directory;
                std::cout << "Введите имя каталога для удаления: ";
                std::cin >> directory;
                try {
                    ftp.removeDirectory(directory);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "8") {
                std::string filePath;
                std::cout << "Введите имя файла для удаления: ";
                std::cin >> filePath;
                try {
                    ftp.deleteFile(filePath);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "9") {
                std::string oldName, newName;
                std::cout << "Введите текущее имя файла/папки: ";
                std::cin >> oldName;
                std::cout << "Введите новое имя: ";
                std::cin >> newName;
                try {
                    ftp.renameFileOrDirectory(oldName, newName);
                } catch (const std::exception& e) {
                    std::cout << e.what() << std::endl;
                }
            } else if (choice == "q") {
                std::cout << "Выход..." << std::endl;
                break;
            } else {
                std::cout << "Неверный выбор. Попробуйте снова." << std::endl;
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "Ошибка: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
