"""
utils/patch.py
==============
Deterministic LSB watermark patch utilities.

All operations are keyed: the same key always produces the same patch,
bit positions, and tiling — guaranteeing encode/decode symmetry.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(key: int) -> np.random.Generator:
    """Return a seeded numpy Generator from an integer key."""
    return np.random.default_rng(key)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_base_patch(patch_size: int, key: int) -> np.ndarray:
    """
    Generate a binary pseudo-random patch.

    Parameters
    ----------
    patch_size : int
        Side length of the square patch (pixels).
    key : int
        Seed used to make the patch deterministic.

    Returns
    -------
    patch : ndarray of shape (patch_size, patch_size), dtype uint8
        Values in {0, 1}.
    """
    rng = _rng(key)
    return rng.integers(0, 2, size=(patch_size, patch_size), dtype=np.uint8)


def get_bit_positions(patch_size: int, n_bits: int, key: int) -> np.ndarray:
    """
    Sample *n_bits* unique (row, col) positions inside the patch.

    Parameters
    ----------
    patch_size : int
        Side length of the patch.
    n_bits : int
        Number of positions to draw (must be ≤ patch_size²).
    key : int
        Seed — same key → same positions.

    Returns
    -------
    positions : ndarray of shape (n_bits, 2), dtype int
        Each row is (row_index, col_index).
    """
    rng = _rng(key + 1)          # offset seed so positions ≠ patch values
    flat = rng.choice(patch_size ** 2, size=n_bits, replace=False)
    return np.stack(np.unravel_index(flat, (patch_size, patch_size)), axis=1)


def embed_watermark(
    base_patch: np.ndarray,
    positions: np.ndarray,
    bits: np.ndarray,
) -> np.ndarray:
    """
    Write watermark bits into a copy of *base_patch* at *positions*.

    Parameters
    ----------
    base_patch : ndarray of shape (P, P), dtype uint8
        Background patch (values in {0, 1}).
    positions : ndarray of shape (n_bits, 2), dtype int
        Pixel positions returned by :func:`get_bit_positions`.
    bits : ndarray of shape (n_bits,), dtype int
        Watermark bits in {0, 1}.  Repeated cyclically if shorter than
        ``len(positions)``.

    Returns
    -------
    patch : ndarray of shape (P, P), dtype uint8
        Patch with watermark bits embedded.
    """
    patch = base_patch.copy()
    bits_tiled = np.resize(bits, len(positions))   # cyclic repeat
    patch[positions[:, 0], positions[:, 1]] = bits_tiled.astype(np.uint8)
    return patch


def upsample_patch(patch: np.ndarray) -> np.ndarray:
    """
    Upsample *patch* by factor 2 (each pixel → 2×2 block).

    Parameters
    ----------
    patch : ndarray of shape (P, P)

    Returns
    -------
    up : ndarray of shape (2P, 2P)
    """
    return np.repeat(np.repeat(patch, 2, axis=0), 2, axis=1)


def tile_patch(
    patch: np.ndarray,
    image_shape: tuple[int, int],
    mode: str = "normal",
) -> np.ndarray:
    """
    Tile *patch* to cover an image of *image_shape*.

    The patch value at (0, 0) is guaranteed to start at image position (0, 0),
    which is required for frequency-domain synchronisation.

    Parameters
    ----------
    patch : ndarray of shape (Ph, Pw)
    image_shape : (H, W)
    mode : {"normal", "symmetric"}
        * ``"normal"``    — wrap padding then crop.
        * ``"symmetric"`` — symmetric padding then crop.

    Returns
    -------
    tiled : ndarray of shape (H, W), same dtype as *patch*
    """
    H, W = image_shape
    Ph, Pw = patch.shape

    if mode == "symmetric":
        pad_h = max(0, H - Ph)
        pad_w = max(0, W - Pw)
        tiled = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="symmetric")
    else:
        # Wrap padding: np.pad repeats the array cyclically
        pad_h = max(0, H - Ph)
        pad_w = max(0, W - Pw)
        tiled = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="wrap")

    return tiled[:H, :W]


def build_patch_image(
    watermark: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
    key: int,
    tile_mode: str = "normal",
) -> np.ndarray:
    """
    Full encode pipeline: base patch → embed → upsample → tile.

    Parameters
    ----------
    watermark : ndarray of shape (n_bits,), dtype int
        Bits in {0, 1} to embed.
    image_shape : (H, W)
    patch_size : int
    key : int
    tile_mode : {"normal", "symmetric"}

    Returns
    -------
    full_patch : ndarray of shape (H, W), dtype uint8
    """
    n_bits = len(watermark)
    base   = generate_base_patch(patch_size, key)
    pos    = get_bit_positions(patch_size, n_bits, key)
    patch  = embed_watermark(base, pos, watermark)
    up     = upsample_patch(patch)
    return tile_patch(up, image_shape, mode=tile_mode)


def build_reference_patch(
    image_shape: tuple[int, int],
    patch_size: int,
    key: int,
    tile_mode: str = "normal",
) -> np.ndarray:
    """
    Reconstruct the *unwatermarked* base patch tiled over *image_shape*.

    Used during decoding to compute the synchronisation reference.

    Parameters
    ----------
    image_shape : (H, W)
    patch_size : int
    key : int
    tile_mode : {"normal", "symmetric"}

    Returns
    -------
    ref : ndarray of shape (H, W), dtype uint8
    """
    base = generate_base_patch(patch_size, key)
    up   = upsample_patch(base)
    return tile_patch(up, image_shape, mode=tile_mode)


# ---------------------------------------------------------------------------
# Extraction + voting
# ---------------------------------------------------------------------------

def extract_bits_from_lsb(
    lsb_image: np.ndarray,
    patch_size: int,
    n_bits: int,
    key: int,
    tile_mode: str = "normal",
):
    """
    Extract watermark bits using cumulative voting with optional symmetric tiling.

    Pipeline
    --------
    1. Convert {0,1} → {-1,+1} (NaNs preserved)
    2. Pad image to full tiles of size (2*patch_size)
    3. Reshape into tiles (up, up, Ty, Tx)
    4. Apply symmetric mirroring if required
    5. Sum all tiles
    6. Sum each 2×2 block → (patch_size, patch_size)
    7. Sample bit positions and vote

    Parameters
    ----------
    lsb_image : (H, W) with {0,1} and NaNs
    patch_size : int
    n_bits : int
    key : int
    tile_mode : {"normal", "symmetric"}

    Returns
    -------
    bits : (n_bits,)
    confidence : float
    """

    H, W = lsb_image.shape
    up = patch_size * 2

    # --- 1. Convert to {-1,+1}, keep NaNs ---
    img = 2.0 * lsb_image.astype(np.float32) - 1.0

    # --- 2. Pad to full tiles ---
    pad_H = (-H) % up
    pad_W = (-W) % up

    img = np.pad(
        img,
        ((0, pad_H), (0, pad_W)),
        mode="constant",
        constant_values=np.nan,
    )

    H2, W2 = img.shape
    Ty, Tx = H2 // up, W2 // up

    # --- 3. Reshape → (up, up, Ty, Tx) ---
    tiles = img.reshape(Ty, up, Tx, up).transpose(1, 3, 0, 2)

    # --- 4. Apply symmetric tiling ---
    if tile_mode == "symmetric":
        # flip tiles on odd rows (vertical flip inside tile)
        tiles[:, :, 1::2, :] = tiles[::-1, :, 1::2, :]

        # flip tiles on odd columns (horizontal flip inside tile)
        tiles[:, :, :, 1::2] = tiles[:, ::-1, :, 1::2]

    # --- 5. Sum tiles ---
    summed = np.nansum(tiles, axis=(2, 3))  # (up, up)

    # --- 6. Sum 2×2 blocks ---
    small = summed.reshape(patch_size, 2, patch_size, 2)
    small = small.sum(axis=(1, 3))  # (patch_size, patch_size)

    # --- 7. Extract bits ---
    pos = get_bit_positions(patch_size, n_bits, key)
    values = small[pos[:, 0], pos[:, 1]]  # sum of votes

    # --- Compute weights (number of valid contributions per bit) ---
    valid_mask = ~np.isnan(img)

    tiles_mask = valid_mask.reshape(Ty, up, Tx, up).transpose(1, 3, 0, 2)
    weights_map = np.sum(tiles_mask, axis=(2, 3))  # (up, up)

    weights_small = weights_map.reshape(patch_size, 2, patch_size, 2)
    weights_small = weights_small.sum(axis=(1, 3))  # (patch_size, patch_size)

    weights = weights_small[pos[:, 0], pos[:, 1]]  # per-bit weights

    # --- Bits ---
    bits = (values > 0).astype(np.uint8)

    # --- Confidence (normalized properly) ---
    weights[weights == 0] = 1  # avoid division by zero
    norm_values = values / weights

    confidence = float(np.mean(np.abs(norm_values)))

    return bits, confidence

# def extract_bits_from_lsb(
#     lsb_image: np.ndarray,
#     image_shape: tuple[int, int],
#     patch_size: int,
#     n_bits: int,
#     key: int,
#     tile_mode: str = "normal",
# ) -> tuple[np.ndarray, float]:
#     """
#     Recover watermark bits from *lsb_image* via majority voting over 2×2 blocks.

#     Each watermark bit is represented by a 2×2 block in the upsampled patch.
#     For every complete tile and every bit, all 4 pixels of the 2×2 block are
#     converted to {-1, +1} and accumulated into a vote.  The sign of the
#     aggregate vote determines the decoded bit; its normalised absolute value
#     is the per-bit confidence.

#     Parameters
#     ----------
#     lsb_image : ndarray of shape (H, W), dtype uint8
#         LSB plane of the (possibly transformed) image.
#     image_shape : (H, W)
#     patch_size : int
#     n_bits : int
#     key : int
#     tile_mode : {"normal", "symmetric"}

#     Returns
#     -------
#     bits : ndarray of shape (n_bits,), dtype uint8
#         Decoded bits in {0, 1}.
#     confidence : float
#         Mean normalised absolute vote across all bits (∈ [0, 1]).
#     """
#     H, W    = image_shape
#     pos     = get_bit_positions(patch_size, n_bits, key)
#     up_size = patch_size * 2          # upsampled tile side length

#     # Upsampled top-left corner of each bit's 2×2 block
#     up_pos = pos * 2                  # shape (n_bits, 2)

#     # Offsets to the four pixels in a 2×2 block
#     dr = np.array([0, 0, 1, 1], dtype=np.intp)
#     dc = np.array([0, 1, 0, 1], dtype=np.intp)

#     votes = np.zeros(n_bits, dtype=np.float64)
#     count = 0

#     for row_start in range(0, H, up_size):
#         for col_start in range(0, W, up_size):
#             # Skip partial tiles at image borders
#             if row_start + up_size > H or col_start + up_size > W:
#                 continue

#             # Row / col indices for all bits' 2×2 blocks in this tile
#             r_idx = up_pos[:, 0:1] + row_start + dr   # (n_bits, 4)
#             c_idx = up_pos[:, 1:2] + col_start + dc   # (n_bits, 4)

#             # Read all 4 pixels per bit, convert {0,1}→{-1,+1}, accumulate
#             block_vals = lsb_image[r_idx, c_idx].astype(np.float64)
#             votes += np.sum(2.0 * block_vals - 1.0, axis=1)
#             count += 1

#     if count == 0:
#         # Fallback: single-tile read without border check
#         r_idx = (up_pos[:, 0:1] % H) + dr   # wrap to image bounds
#         c_idx = (up_pos[:, 1:2] % W) + dc
#         r_idx = np.clip(r_idx, 0, H - 1)
#         c_idx = np.clip(c_idx, 0, W - 1)
#         block_vals = lsb_image[r_idx, c_idx].astype(np.float64)
#         votes = np.sum(2.0 * block_vals - 1.0, axis=1)
#         count = 1

#     # Each tile contributes 4 values per bit
#     norm_votes = votes / (count * 4)
#     bits       = (norm_votes > 0).astype(np.uint8)
#     confidence = float(np.mean(np.abs(norm_votes)))
#     return bits, confidence
