CC = gcc
CFLAGS = -Wall

all: server client

server: server.c
	$(CC) $(CFLAGS) -o server server.c

client: client.c
	$(CC) $(CFLAGS) -o client client.c

run:
	./server

clean:
	rm -f server client
	rm -f log.txt
	
# ./client 127.0.0.1 55737 2 3