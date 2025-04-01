// server.c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <pthread.h>

pthread_mutex_t file_mutex = PTHREAD_MUTEX_INITIALIZER;

void *handle_client(void *arg)
{
    int client_sock = *(int *)arg;
    free(arg);

    FILE *file;
    char buf[100];
    int msgLength;

    while ((msgLength = recv(client_sock, buf, 100 - 1, 0)) > 0)
    {
        buf[msgLength] = '\0';
        printf("SERVER: Получено сообщение: %s\n", buf);

        pthread_mutex_lock(&file_mutex);
        file = fopen("log.txt", "a");
        if (file)
        {
            fprintf(file, "%s\n", buf);
            fclose(file);
        }
        pthread_mutex_unlock(&file_mutex);

        sleep(atoi(buf));
    }
    close(client_sock);
    return NULL;
}

int main()
{
    int sockMain, *sockClient;
    struct sockaddr_in servAddr, clientAddr;
    socklen_t clientLen = sizeof(clientAddr);

    if ((sockMain = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("Ошибка создания сокета");
        exit(1);
    }

    memset(&servAddr, 0, sizeof(servAddr));
    servAddr.sin_family = AF_INET;
    servAddr.sin_addr.s_addr = htonl(INADDR_ANY);
    servAddr.sin_port = htons(0);

    if (bind(sockMain, (struct sockaddr *)&servAddr, sizeof(servAddr)) < 0)
    {
        perror("Ошибка привязки сокета");
        exit(1);
    }

    socklen_t len = sizeof(servAddr);
    if (getsockname(sockMain, (struct sockaddr *)&servAddr, &len) == -1)
    {
        perror("Ошибка getsockname");
        exit(1);
    }
    printf("Сервер запущен на порту %d\n", ntohs(servAddr.sin_port));

    listen(sockMain, 5);

    while (1)
    {
        sockClient = malloc(sizeof(int));
        if ((*sockClient = accept(sockMain, (struct sockaddr *)&clientAddr, &clientLen)) < 0)
        {
            perror("Ошибка accept");
            free(sockClient);
            continue;
        }

        pthread_t thread;
        pthread_create(&thread, NULL, handle_client, sockClient);
        pthread_detach(thread);
    }
    close(sockMain);
    return 0;
}
