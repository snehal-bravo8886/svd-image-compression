"""
P3: RGB image compression using rank-k SVD.
Idea: a colour image = 3 matrices (R, G, B). Compress each one separately
with rank-k SVD, clip to 0-255, stack back, convert to uint8.

Install:  pip install numpy matplotlib pillow
Run:      python rgb_compression.py            (expects photo.jpg in same folder)
"""
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


# ---- SVD source -------------------------------------------------------
# For now we use NumPy's SVD. When P1 finishes my_svd, import it and
# change USE_MY_SVD to True. It must return U, S, Vt like np.linalg.svd.
USE_MY_SVD = False
# from svd_engine import my_svd      # <- P1's file (uncomment later)


def get_svd(A):
    if USE_MY_SVD:
        return my_svd(A)  # noqa: F821
    return np.linalg.svd(A, full_matrices=False)


# ---- Core functions ---------------------------------------------------
def rank_k(U, S, Vt, k):
    """Rebuild a matrix from the top k layers."""
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]


def compress_rgb(img_array, k_r, k_g=None, k_b=None):
    """
    img_array: H x W x 3 float array.
    Compresses R, G, B separately. If only k_r is given, all channels use it.
    Returns an H x W x 3 uint8 image.
    """
    ks = [k_r, k_r if k_g is None else k_g, k_r if k_b is None else k_b]
    channels = []
    for c in range(3):
        A = img_array[:, :, c]
        U, S, Vt = get_svd(A)
        A_k = rank_k(U, S, Vt, ks[c])
        A_k = np.clip(A_k, 0, 255)          # SVD can overshoot 0-255
        channels.append(A_k)
    out = np.stack(channels, axis=2)        # stack back into H x W x 3
    return out.astype(np.uint8)             # convert to valid image type


def storage_rgb(m, n, k):
    """Numbers stored: 3 channels x k(m+n+1). Original is 3*m*n."""
    return 3 * k * (m + n + 1), 3 * m * n


# ---- Demo -------------------------------------------------------------
if __name__ == "__main__":
    img = Image.open("photo.jpg").convert("RGB").resize((512, 512))
    A = np.array(img, dtype=float)
    print("RGB array shape:", A.shape)       # (512, 512, 3)

    ks = [5, 20, 50, 100]
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    axes[0].imshow(img)
    axes[0].set_title("Original")
    for ax, k in zip(axes[1:], ks):
        ax.imshow(compress_rgb(A, k))
        ax.set_title(f"k = {k}")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("rgb_comparison.png", dpi=150)
    plt.show()

    # Storage info for the report
    for k in ks:
        used, orig = storage_rgb(512, 512, k)
        print(f"k={k:3d}: stores {used:,} vs original {orig:,}  ({used/orig:.1%})")

    # Viva demo: different k per channel (e.g. keep more detail in green)
    mixed = compress_rgb(A, k_r=10, k_g=50, k_b=10)
    plt.imshow(mixed)
    plt.title("k_R=10, k_G=50, k_B=10")
    plt.axis("off")
    plt.savefig("rgb_mixed_k.png", dpi=150)
    plt.show()
