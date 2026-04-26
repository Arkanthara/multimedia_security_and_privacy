"""
utils/patch.py
==============
Deterministic watermark patch utilities.

All operations are keyed: the same key always produces the same patch,
bit positions, and tiling — guaranteeing encode/decode symmetry.

The *upsampling factor* (``upsample_factor``, default 2) controls how many
pixels each logical patch cell occupies.  A cell of 2×2 pixels is the
classic choice; 3×3 gives denser embedding at the cost of more pixels per
bit.  The parameter must be identical at encode and decode time.

Public API
----------
LSB helpers:
    generate_base_patch, get_bit_positions, embed_watermark,
    upsample_patch, tile_patch, build_patch_image,
    build_reference_patch, extract_bits_from_lsb

Spatial helpers:
    build_patch_image_bipolar, build_reference_patch_bipolar,
    extract_bits_from_spatial

Performance notes
-----------------
* ``tile_patch`` uses ``np.tile`` for "normal" (wrap) mode instead of
  ``np.pad(mode="wrap")``.  ``np.tile`` fills the output in a single pass
  without the bookkeeping overhead of the padding API.

* ``build_patch_image_bipolar`` and ``build_reference_patch_bipolar`` are
  inlined: they no longer call ``build_patch_image`` / ``build_reference_patch``
  as intermediaries.  This eliminates one extra function-call level and one
  intermediate uint8 allocation, going straight to float32 output.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _rng(key: int) -> np.random.Generator:
    """Return a seeded numpy Generator from an integer key."""
    return np.random.default_rng(key)


# ---------------------------------------------------------------------------
# Public API — patch construction
# ---------------------------------------------------------------------------

def generate_base_patch(patch_size: int, key: int) -> np.ndarray:
    """
    Generate a binary pseudo-random patch.

    Parameters
    ----------
    patch_size : int
        Side length of the square patch (logical pixels).
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
    rng  = _rng(key + 1)          # offset seed so positions ≠ patch values
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
    positions : ndarray of shape (n_bits, 2), dtype int
    bits : ndarray of shape (n_bits,), dtype int
        Watermark bits in {0, 1}.  Repeated cyclically if shorter.

    Returns
    -------
    patch : ndarray of shape (P, P), dtype uint8
    """
    patch      = base_patch.copy()
    bits_tiled = np.resize(bits, len(positions))
    patch[positions[:, 0], positions[:, 1]] = bits_tiled.astype(np.uint8)
    return patch


def upsample_patch(patch: np.ndarray, factor: int = 2) -> np.ndarray:
    """
    Upsample *patch* by *factor* (each logical pixel → ``factor × factor``
    image pixels).

    Parameters
    ----------
    patch : ndarray of shape (P, P)
    factor : int
        Upsampling factor (default 2).  Must be ≥ 1.

    Returns
    -------
    up : ndarray of shape (P*factor, P*factor)
    """
    return np.repeat(np.repeat(patch, factor, axis=0), factor, axis=1)


def tile_patch(
    patch: np.ndarray,
    image_shape: tuple[int, int],
    mode: str = "normal",
) -> np.ndarray:
    """
    Tile *patch* to cover an image of *image_shape*.

    Parameters
    ----------
    patch : ndarray of shape (Ph, Pw)
    image_shape : (H, W)
    mode : {"normal", "symmetric"}

    Returns
    -------
    tiled : ndarray of shape (H, W), same dtype as *patch*

    Notes
    -----
    For "normal" (wrap) mode ``np.tile`` is used instead of
    ``np.pad(mode="wrap")``.  ``np.tile`` fills the result in a single pass
    with no padding-API overhead; ``np.pad`` with mode="wrap" adds bookkeeping
    that is measurable for large images.
    """
    H, W   = image_shape
    Ph, Pw = patch.shape

    if mode == "symmetric":
        pad_h = max(0, H - Ph)
        pad_w = max(0, W - Pw)
        tiled = np.pad(patch, ((0, pad_h), (0, pad_w)), mode="symmetric")
    else:
        # ceil-division repeat counts, then crop to exact size
        reps_h = -(-H // Ph)   # equivalent to math.ceil(H / Ph)
        reps_w = -(-W // Pw)
        tiled  = np.tile(patch, (reps_h, reps_w))

    return tiled[:H, :W]


def build_patch_image(
    watermark: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
    key: int,
    tile_mode: str = "normal",
    upsample_factor: int = 2,
) -> np.ndarray:
    """
    Full encode pipeline: base patch → embed watermark → upsample → tile.

    Parameters
    ----------
    watermark : ndarray of shape (n_bits,), dtype int
        Bits in {0, 1} to embed.
    image_shape : (H, W)
    patch_size : int
        Logical patch side length (before upsampling).
    key : int
    tile_mode : {"normal", "symmetric"}
    upsample_factor : int
        Each logical patch pixel expands to a ``factor × factor`` block.

    Returns
    -------
    full_patch : ndarray of shape (H, W), dtype uint8
        Values in {0, 1}.
    """
    n_bits = len(watermark)
    base   = generate_base_patch(patch_size, key)
    pos    = get_bit_positions(patch_size, n_bits, key)
    patch  = embed_watermark(base, pos, watermark)
    up     = upsample_patch(patch, factor=upsample_factor)
    return tile_patch(up, image_shape, mode=tile_mode)


def build_reference_patch(
    image_shape: tuple[int, int],
    patch_size: int,
    key: int,
    tile_mode: str = "normal",
    upsample_factor: int = 2,
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
    upsample_factor : int

    Returns
    -------
    ref : ndarray of shape (H, W), dtype uint8
        Values in {0, 1}.
    """
    base = generate_base_patch(patch_size, key)
    up   = upsample_patch(base, factor=upsample_factor)
    return tile_patch(up, image_shape, mode=tile_mode)


# ---------------------------------------------------------------------------
# Spatial watermarking helpers
# ---------------------------------------------------------------------------

def build_patch_image_bipolar(
    watermark: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
    key: int,
    tile_mode: str = "symmetric",
    upsample_factor: int = 2,
) -> np.ndarray:
    """
    Build a bipolar ({-1, +1}) watermark patch tiled over *image_shape*.

    Inlined version of ``build_patch_image`` + bipolar conversion: avoids
    one extra call level and the intermediate uint8 array allocation.

    Parameters
    ----------
    watermark : ndarray of shape (n_bits,), dtype int
    image_shape : (H, W)
    patch_size : int
    key : int
    tile_mode : {"normal", "symmetric"}
    upsample_factor : int

    Returns
    -------
    full_patch : ndarray of shape (H, W), dtype float32
        Values in {-1.0, +1.0}.
    """
    n_bits = len(watermark)
    base   = generate_base_patch(patch_size, key)
    pos    = get_bit_positions(patch_size, n_bits, key)
    patch  = embed_watermark(base, pos, watermark)
    up     = upsample_patch(patch, factor=upsample_factor)
    tiled  = tile_patch(up, image_shape, mode=tile_mode)
    # {0,1} → {-1.0, +1.0} in float32, single allocation
    return np.where(tiled, np.float32(1.0), np.float32(-1.0))


def build_reference_patch_bipolar(
    image_shape: tuple[int, int],
    patch_size: int,
    key: int,
    tile_mode: str = "symmetric",
    upsample_factor: int = 2,
) -> np.ndarray:
    """
    Build a bipolar ({-1, +1}) reference patch (no watermark bits) tiled over
    *image_shape*.

    Inlined version of ``build_reference_patch`` + bipolar conversion.

    Parameters
    ----------
    image_shape : (H, W)
    patch_size : int
    key : int
    tile_mode : {"normal", "symmetric"}
    upsample_factor : int

    Returns
    -------
    ref : ndarray of shape (H, W), dtype float32
        Values in {-1.0, +1.0}.
    """
    base  = generate_base_patch(patch_size, key)
    up    = upsample_patch(base, factor=upsample_factor)
    tiled = tile_patch(up, image_shape, mode=tile_mode)
    return np.where(tiled, np.float32(1.0), np.float32(-1.0))


# ---------------------------------------------------------------------------
# Extraction + voting — LSB version
# ---------------------------------------------------------------------------

def extract_bits_from_lsb(
    lsb_image: np.ndarray,
    patch_size: int,
    n_bits: int,
    key: int,
    tile_mode: str = "normal",
    upsample_factor: int = 2,
):
    """
    Extract watermark bits using cumulative voting with optional symmetric tiling.

    Pipeline
    --------
    1. Convert {0,1} → {-1,+1} (NaNs preserved)
    2. Pad image to full tiles of size (patch_size * upsample_factor)
    3. Reshape into tiles
    4. Apply symmetric mirroring correction if required
    5. Sum all tiles
    6. Sum each ``upsample_factor × upsample_factor`` block → (patch_size, patch_size)
    7. Sample bit positions and vote

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W), values in {0, 1} (NaNs allowed)
    patch_size : int
    n_bits : int
    key : int
    tile_mode : {"normal", "symmetric"}
    upsample_factor : int

    Returns
    -------
    bits : ndarray of shape (n_bits,), dtype uint8
    confidence : float
    """
    H, W = lsb_image.shape
    up   = patch_size * upsample_factor

    img = 2.0 * lsb_image.astype(np.float32) - 1.0

    # Pad to full tiles
    pad_H = (-H) % up
    pad_W = (-W) % up
    img   = np.pad(img, ((0, pad_H), (0, pad_W)), mode="constant",
                   constant_values=np.nan)

    H2, W2 = img.shape
    Ty, Tx  = H2 // up, W2 // up

    tiles = img.reshape(Ty, up, Tx, up).transpose(1, 3, 0, 2)   # (up, up, Ty, Tx)

    if tile_mode == "symmetric":
        tiles[:, :, 1::2, :] = tiles[::-1, :, 1::2, :]
        tiles[:, :, :, 1::2] = tiles[:, ::-1, :, 1::2]

    summed = np.nansum(tiles, axis=(2, 3))    # (up, up)

    # Accumulate upsample_factor × upsample_factor pixel blocks
    small = summed.reshape(
        patch_size, upsample_factor, patch_size, upsample_factor
    ).sum(axis=(1, 3))                         # (patch_size, patch_size)

    pos    = get_bit_positions(patch_size, n_bits, key)
    values = small[pos[:, 0], pos[:, 1]]

    # Confidence via weight map
    valid_mask    = ~np.isnan(img)
    tiles_mask    = valid_mask.reshape(Ty, up, Tx, up).transpose(1, 3, 0, 2)
    weights_map   = np.sum(tiles_mask, axis=(2, 3))
    weights_small = weights_map.reshape(
        patch_size, upsample_factor, patch_size, upsample_factor
    ).sum(axis=(1, 3))
    weights = weights_small[pos[:, 0], pos[:, 1]]

    bits = (values > 0).astype(np.uint8)

    weights[weights == 0] = 1
    norm_values = values / weights
    confidence  = float(np.mean(np.abs(norm_values)))

    return bits, confidence


# ---------------------------------------------------------------------------
# Extraction + voting — spatial version
# ---------------------------------------------------------------------------

def extract_bits_from_spatial(
    residual: np.ndarray,
    patch_size: int,
    n_bits: int,
    key: int,
    tile_mode: str = "symmetric",
    upsample_factor: int = 2,
) -> tuple[np.ndarray, float]:
    """
    Extract watermark bits from a spatial-domain residual signal.

    The residual (``watermarked_image − denoised_image``) carries the embedded
    bipolar watermark.  This function accumulates all tiles of the residual and
    reads off bit decisions at the keyed positions.

    When called after :func:`~utils.sync.synchronise`, *residual* is already
    the aligned, accumulated block of shape ``(patch_size * upsample_factor,
    patch_size * upsample_factor)``.  In that case there is exactly one tile
    and the accumulation is a no-op, which is the intended use.

    Pipeline
    --------
    1. Pad residual to full tiles of size ``patch_size * upsample_factor``
    2. Reshape into tiles, undo symmetric mirroring if required
    3. Sum all tiles → accumulated patch ``(up, up)``
    4. Sum each ``upsample_factor × upsample_factor`` block → ``(patch_size, patch_size)``
    5. Sample keyed bit positions; threshold at 0

    Parameters
    ----------
    residual : ndarray of shape (H, W), float
        Per-pixel residual.  Shape ``(up, up)`` when called post-synchronise.
    patch_size : int
    n_bits : int
    key : int
    tile_mode : {"normal", "symmetric"}
    upsample_factor : int

    Returns
    -------
    bits : ndarray of shape (n_bits,), dtype uint8
    confidence : float
    """
    H, W = residual.shape
    up   = patch_size * upsample_factor

    img = residual.astype(np.float32)

    # Pad to full tiles (zero padding)
    pad_H = (-H) % up
    pad_W = (-W) % up
    img   = np.pad(img, ((0, pad_H), (0, pad_W)), mode="constant",
                   constant_values=0.0)

    H2, W2 = img.shape
    Ty, Tx  = H2 // up, W2 // up

    tiles = img.reshape(Ty, up, Tx, up).transpose(1, 3, 0, 2)   # (up, up, Ty, Tx)

    if tile_mode == "symmetric":
        tiles[:, :, 1::2, :] = tiles[::-1, :, 1::2, :]
        tiles[:, :, :, 1::2] = tiles[:, ::-1, :, 1::2]

    summed = np.sum(tiles, axis=(2, 3))    # (up, up)

    # Sum upsample_factor × upsample_factor pixel blocks into logical patch
    small = summed.reshape(
        patch_size, upsample_factor, patch_size, upsample_factor
    ).sum(axis=(1, 3))                     # (patch_size, patch_size)

    pos    = get_bit_positions(patch_size, n_bits, key)
    values = small[pos[:, 0], pos[:, 1]]

    bits = (values > 0).astype(np.uint8)

    peak = np.abs(values).max()
    norm = values / (peak + 1e-8)
    confidence = float(np.mean(np.abs(norm)))

    return bits, confidence
