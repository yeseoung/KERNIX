CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -I./include

test: test/test.c include/kernix.h
	$(CC) $(CFLAGS) test/test.c -o test/test

run: test
	./test/test

clean:
	rm -f test/test