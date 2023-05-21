CFLAGS=-c -Wall -I./src -g3
LDFLAGS=
SOURCES=src/main.c src/mystring.c src/cyg2win.c
OBJECTS=$(SOURCES:.c=.o)
EXECUTABLE=bin/app

all: $(SOURCES) $(EXECUTABLE)
	
$(EXECUTABLE): $(OBJECTS) 
	$(CC) $(LDFLAGS) $(OBJECTS) -o $@

.c.o:
	$(CC) $(CFLAGS) $< -o $@

clean:
	rm src/*.o bin/*
