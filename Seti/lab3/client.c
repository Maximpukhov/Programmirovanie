#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>

int main(int argc, char *argv[])
{
    if (argc != 5)
    {
        printf("Использование: %s <IP-адрес> <порт> <число i> <количество повторений>\n", argv[0]);
        exit(1);
    }

    int sock, i, num_repeats = atoi(argv[4]);
    struct sockaddr_in servAddr;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("Ошибка создания сокета");
        exit(1);
    }

    memset(&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(atoi(argv[2]));

    if (inet_pton(AF_INET, argv[1], &servAddr.sin_addr) <= 0)
    {
        perror("Ошибка преобразования IP-адреса");
        exit(1);
    }

    if (connect(sock, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
    {
        perror("Ошибка подключения к серверу");
        exit(1);
    }

    for (i = 0; i < num_repeats; i++)
    {
        char message[10];
        sprintf(message, "%d", atoi(argv[3]));
        send(sock, message, strlen(message), 0);
        sleep(atoi(argv[3]));
    }

    close(sock);
    return 0;
}
