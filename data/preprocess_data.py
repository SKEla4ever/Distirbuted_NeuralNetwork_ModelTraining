# This python code is used to preprocess the MNIST dataset and write it into a binary format 
# that can be read by the C server and worker programs. It loads the MNIST data, normalizes it, 
# flattens the images, and splits the training set into multiple shards. 
# Each shard is written to disk in a specific binary format that includes metadata about the number 
# of samples, features, and classes, followed by the feature data and labels. The script also prints 
# out some information about the dimensions of the model for reference when implementing the C code.

import os
import math
import struct
import numpy as np

# Try TensorFlow Keras first
try:
    from tensorflow.keras.datasets import mnist
except ImportError as e:
    raise ImportError(
        "TensorFlow is required for this script.\n"
        "Install it with: pip install tensorflow"
    ) from e


def write_shard(path: str, X: np.ndarray, y: np.ndarray, n_classes: int) -> None:
    """
    Write one shard in the format expected by worker.c:

    int32: n_samples
    int32: n_features
    int32: n_classes
    float32[n_samples * n_features]: features
    int32[n_samples]: labels
    """
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    if len(X) != len(y):
        raise ValueError("X and y must have the same number of samples")

    n_samples, n_features = X.shape

    # Enforce exact dtypes expected on disk
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int32)

    with open(path, "wb") as f:
        # metadata: 3 int32 values
        f.write(struct.pack("iii", n_samples, n_features, n_classes))
        # features: row-major float32
        f.write(X.tobytes(order="C"))
        # labels: int32
        f.write(y.tobytes(order="C"))

    print(
        f"Wrote {path}: "
        f"{n_samples} samples, {n_features} features, {n_classes} classes"
    )


def split_into_equal_shards(X: np.ndarray, y: np.ndarray, num_shards: int):
    """
    Split X, y into num_shards nearly equal contiguous parts.
    """
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")

    n = len(X)
    shard_sizes = [n // num_shards] * num_shards
    for i in range(n % num_shards):
        shard_sizes[i] += 1

    shards = []
    start = 0
    for size in shard_sizes:
        end = start + size
        shards.append((X[start:end], y[start:end]))
        start = end
    return shards


def main():
    output_dir = "mnist_shards"
    num_train_shards = 5
    shuffle_train = True
    random_seed = 42

    os.makedirs(output_dir, exist_ok=True)

    # Load MNIST
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Shapes before preprocessing:
    # x_train: (60000, 28, 28)
    # y_train: (60000,)
    # x_test:  (10000, 28, 28)
    # y_test:  (10000,)

    # Normalize to [0, 1] and flatten to (N, 784)
    X_train = x_train.astype(np.float32) / 255.0
    X_test = x_test.astype(np.float32) / 255.0

    X_train = X_train.reshape(X_train.shape[0], -1)  # (60000, 784)
    X_test = X_test.reshape(X_test.shape[0], -1)     # (10000, 784)

    y_train = y_train.astype(np.int32)
    y_test = y_test.astype(np.int32)

    n_classes = 10

    # Optional shuffle before sharding
    if shuffle_train:
        rng = np.random.default_rng(random_seed)
        perm = rng.permutation(len(X_train))
        X_train = X_train[perm]
        y_train = y_train[perm]

    # Split training set into equal shards
    train_shards = split_into_equal_shards(X_train, y_train, num_train_shards)

    # Write train shards
    for i, (Xs, ys) in enumerate(train_shards):
        shard_path = os.path.join(output_dir, f"train_shard_{i}.bin")
        write_shard(shard_path, Xs, ys, n_classes)

    # Write a full test shard too
    test_path = os.path.join(output_dir, "test_full.bin")
    write_shard(test_path, X_test, y_test, n_classes)

    print("\nDone.")
    print(f"Training shards written to: {output_dir}/train_shard_*.bin")
    print(f"Test shard written to:      {output_dir}/test_full.bin")

    # Helpful dimension info for your C programs
    input_size = X_train.shape[1]   # 784
    output_size = n_classes         # 10

    print("\nModel dimension examples:")
    for hidden_size in [32, 64, 128]:
        dim = input_size * hidden_size + hidden_size + hidden_size * output_size + output_size
        print(
            f"  hidden_size={hidden_size:<3} -> "
            f"input_size={input_size}, output_size={output_size}, dim={dim}"
        )


if __name__ == "__main__":
    main()