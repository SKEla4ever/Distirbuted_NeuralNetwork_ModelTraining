import numpy as np

for path in ["demo/1.bin", "demo/5.bin", "demo/9.bin"]:
    x = np.fromfile(path, dtype=np.float32)
    print(path)
    print("shape:", x.shape)
    print("sum:", float(x.sum()))
    print("nonzero(>1e-6):", int((x > 1e-6).sum()))
    print("first 20:", x[:20])
    print()