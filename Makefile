CC      = gcc
CFLAGS  = -Wall -O3 -march=native -ffast-math
LDFLAGS = -lm
OUT     = gpt

.DEFAULT_GOAL := single

single:
	$(CC) $(CFLAGS) run_model.c -o $(OUT) $(LDFLAGS)

multi:
	$(CC) $(CFLAGS) train_parallel.c -o $(OUT) $(LDFLAGS) -pthread

clean:
	rm -f $(OUT)

.PHONY: single multi clean
