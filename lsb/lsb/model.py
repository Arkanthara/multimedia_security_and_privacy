"""
lsb/model.py
============
LSB watermark — encode and decode entry points.

Encode pipeline
---------------
    image  →  build patch  →  replace LSB  →  watermarked image

Decode pipeline (sync_enabled=True)
------------------------------------
    Step 1  direct decode               → return if confidence ≥ threshold
    Step 2  rotation/scale correction   → return if confidence ≥ threshold
    Step 3  translation correction      → return best result
"""

from __future__ import annotations

import numpy as np
from skimage.util import img_as_ubyte

from utils.patch import (
    build_patch_image,
    build_reference_patch,
    extract_bits_from_lsb,
)
from utils.sync import synchronise


# Default confidence threshold for early exit
CONFIDENCE_THRESHOLD = 0.8


# ---------------------------------------------------------------------------
# Encode
# ---------------------------------------------------------------------------

def encode(
    image: np.ndarray,
    watermark: np.ndarray,
    patch_size: int,
    key: int,
    tile_mode: str = "normal",
) -> np.ndarray:
    """
    Embed *watermark* bits into the LSB plane of *image*.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, C), any dtype
        Input image.  Converted to uint8 internally.
    watermark : ndarray of shape (n_bits,), dtype int
        Bits in {0, 1} to embed.
    patch_size : int
        Side length of the square base patch (before ×2 upsample).
    key : int
        Secret key — controls patch content and bit positions.
    tile_mode : {"normal", "symmetric"}
        How the patch is tiled over the image.

    Returns
    -------
    watermarked : ndarray, same shape as *image*, dtype uint8
        Image with watermark bits written into the LSB.
    """
    img_u8  = img_as_ubyte(image)
    H, W    = img_u8.shape[:2]

    patch = build_patch_image(watermark, (H, W), patch_size, key, tile_mode)

    if img_u8.ndim == 3:
        # Apply to luminance channel (green, index 1) only for colour images
        result         = img_u8.copy()
        result[..., 1] = (img_u8[..., 1] & np.uint8(0xFE)) | patch
    else:
        result = (img_u8 & np.uint8(0xFE)) | patch

    return result


# ---------------------------------------------------------------------------
# Internal: single-pass decode
# ---------------------------------------------------------------------------

def _decode_pass(
    lsb_plane: np.ndarray,
    image_shape: tuple[int, int],
    patch_size: int,
    n_bits: int,
    key: int,
    tile_mode: str,
) -> tuple[np.ndarray, float]:
    """
    Extract bits + confidence from a (possibly pre-aligned) *lsb_plane*.

    Parameters
    ----------
    lsb_plane : ndarray of shape (H, W), uint8
    image_shape : (H, W)
    patch_size : int
    n_bits : int
    key : int
    tile_mode : str

    Returns
    -------
    bits : ndarray of shape (n_bits,), uint8
    confidence : float
    """
    return extract_bits_from_lsb(
        lsb_plane, image_shape, patch_size, n_bits, key, tile_mode
    )


# ---------------------------------------------------------------------------
# Decode
# ---------------------------------------------------------------------------

def decode(
    image: np.ndarray,
    n_bits: int,
    patch_size: int,
    key: int,
    tile_mode: str = "normal",
    sync_enabled: bool = True,
    upsample_factor_rs: int = 1,
    upsample_factor_t: int = 10,
) -> tuple[np.ndarray, float]:
    """
    Recover watermark bits from *image*.

    When *sync_enabled* is ``True`` the decoder tries three alignments
    in order, returning early on the first whose confidence exceeds
    ``CONFIDENCE_THRESHOLD``, then falls back to the best overall result.

    Parameters
    ----------
    image : ndarray of shape (H, W) or (H, W, C)
        Possibly transformed watermarked image.
    n_bits : int
        Number of watermark bits to recover.
    patch_size : int
        Must match the value used in :func:`encode`.
    key : int
        Must match the value used in :func:`encode`.
    tile_mode : {"normal", "symmetric"}
    sync_enabled : bool
        Enable rotation/scale and translation correction.
    upsample_factor_rs : int
        Sub-pixel factor for rotation/scale phase correlation.
    upsample_factor_t : int
        Sub-pixel factor for translation phase correlation.

    Returns
    -------
    bits : ndarray of shape (n_bits,), dtype uint8
        Decoded bits in {0, 1}.
    confidence : float
        Reliability score ∈ [0, 1].  Values ≥ 0.8 are considered robust.
    """
    img_u8 = img_as_ubyte(image)
    H, W   = img_u8.shape[:2]

    # Extract LSB plane (green channel for colour images)
    channel = img_u8[..., 1] if img_u8.ndim == 3 else img_u8
    lsb     = (channel & np.uint8(1)).astype(np.uint8)

    # -----------------------------------------------------------------------
    # Step 1 — direct decode (no alignment)
    # -----------------------------------------------------------------------
    bits, conf = _decode_pass(lsb, (H, W), patch_size, n_bits, key, tile_mode)
    if not sync_enabled or conf >= CONFIDENCE_THRESHOLD:
        return bits, conf

    best_bits, best_conf = bits, conf

    # -----------------------------------------------------------------------
    # Step 2 — rotation + scale + translation correction
    # -----------------------------------------------------------------------
    reference     = build_reference_patch((H, W), patch_size, key, tile_mode)
    lsb_rs, _     = synchronise(
        lsb, reference,
        upsample_factor_rs=upsample_factor_rs,
        upsample_factor_t=upsample_factor_t,
    )

    bits_rs, conf_rs = _decode_pass(
        lsb_rs, (H, W), patch_size, n_bits, key, tile_mode
    )
    if conf_rs > best_conf:
        best_bits, best_conf = bits_rs, conf_rs
    if best_conf >= CONFIDENCE_THRESHOLD:
        return best_bits, best_conf

    return best_bits, best_conf
