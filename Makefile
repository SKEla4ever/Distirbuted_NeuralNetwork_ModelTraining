CC = gcc
CFLAGS = -Wall -Wextra -std=c11 -g
LDFLAGS = -lm

all: server_app worker_app predict_app

server_app: server/server.c
	$(CC) $(CFLAGS) server/server.c -o server_app $(LDFLAGS)

worker_app: worker/worker.c common/neural_network.c
	$(CC) $(CFLAGS) worker/worker.c common/neural_network.c -o worker_app $(LDFLAGS)

clean:
	rm -rf server_app worker_app *.o *.bin predict_app *.dSYM

run_server:
	./server_app 6666 5 50890 0.1 100

run_worker0: worker_app
	./worker_app 127.0.0.1 6666 64 mnist_shards/train_shard_0.bin

run_worker1: worker_app
	./worker_app 127.0.0.1 6666 64 mnist_shards/train_shard_1.bin

run_worker2: worker_app
	./worker_app 127.0.0.1 6666 64 mnist_shards/train_shard_2.bin

run_worker3: worker_app
	./worker_app 127.0.0.1 6666 64 mnist_shards/train_shard_3.bin

run_worker4: worker_app
	./worker_app 127.0.0.1 6666 64 mnist_shards/train_shard_4.bin

predict_app: demo/predict_from_vector.c common/neural_network.c
	$(CC) $(CFLAGS) demo/predict_from_vector.c common/neural_network.c -o predict_app $(LDFLAGS)