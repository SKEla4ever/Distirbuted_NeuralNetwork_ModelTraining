import sys
import numpy as np
from PIL import Image, ImageOps

MNIST_SIZE = 28
FIT_SIZE = 20  # scale digit to fit inside 20x20, then center in 28x28


def preprocess_image_to_vector(image_path: str) -> np.ndarray:
    img = Image.open(image_path).convert("L")
    arr = np.array(img, dtype=np.uint8)

    # If background is mostly white, invert so digit becomes bright on dark background
    if arr.mean() > 127:
        arr = 255 - arr

    # Find bounding box of nontrivial pixels
    ys, xs = np.where(arr > 30)
    if len(xs) == 0 or len(ys) == 0:
        raise ValueError("No visible digit found in the image.")

    x0, x1 = xs.min(), xs.max() + 1
    y0, y1 = ys.min(), ys.max() + 1
    cropped = arr[y0:y1, x0:x1]

    h, w = cropped.shape

    # Scale to fit inside FIT_SIZE x FIT_SIZE while preserving aspect ratio
    scale = min(FIT_SIZE / w, FIT_SIZE / h)
    new_w = max(1, round(w * scale))
    new_h = max(1, round(h * scale))

    cropped_img = Image.fromarray(cropped, mode="L")
    resized = cropped_img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # Place in center of 28x28 canvas
    canvas = Image.new("L", (MNIST_SIZE, MNIST_SIZE), color=0)
    off_x = (MNIST_SIZE - new_w) // 2
    off_y = (MNIST_SIZE - new_h) // 2
    canvas.paste(resized, (off_x, off_y))

    vec = np.array(canvas, dtype=np.float32) / 255.0
    return vec.reshape(-1)  # length 784


def main():
    if len(sys.argv) != 3:
        print("Usage: python3 preprocess_single_image.py <input_image> <output_vector.bin>")
        sys.exit(1)

    input_image = sys.argv[1]
    output_bin = sys.argv[2]

    vec = preprocess_image_to_vector(input_image)

    if vec.shape[0] != 784:
        raise ValueError(f"Expected 784 features, got {vec.shape[0]}")

    vec.astype(np.float32).tofile(output_bin)
    print(f"Saved {output_bin} with {vec.shape[0]} float32 values")


if __name__ == "__main__":
    main()