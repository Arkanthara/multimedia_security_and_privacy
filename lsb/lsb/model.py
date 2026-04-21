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
    extract_bits_from_lsb,
)
from utils.sync import synchronise


class WatermarkModel:
    def __init__(
		self,
		message_length: int = 32,
        n_repeats: int = 3,
		psnr_threshold: float = 30.0,
		max_encode_time: float = 5.0,
		max_decode_time: float = 1.0,
        patch_size: int = 32,
        key: int = 42,
        tile_mode: str = "symmetric",
        confidence_threshold: float = 0.8,
        sync_enabled: bool = True,
        upsample_factor_rs: int = 10,
        upsample_factor_t: int = 10,

	):
        self.message_length = message_length
        self.n_repeats = n_repeats
        self.psnr_threshold = psnr_threshold
        self.max_encode_time = max_encode_time
        self.max_decode_time = max_decode_time
        self.patch_size = patch_size
        self.key = key
        self.tile_mode = tile_mode
        self.confidence_threshold = confidence_threshold
        self.sync_enabled = sync_enabled
        self.upsample_factor_rs = upsample_factor_rs
        self.upsample_factor_t = upsample_factor_t
    # ---------------------------------------------------------------------------
    # Encode
    # ---------------------------------------------------------------------------

    def encode(
        self,
        image: np.ndarray,
        watermark: np.ndarray,
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

        watermark = np.asarray(watermark, dtype=np.uint8)
        watermark = np.tile(watermark, self.n_repeats)

        patch = build_patch_image(watermark, (H, W), self.patch_size, self.key, self.tile_mode)

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
        self,
        lsb_plane: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """
        Extract bits + confidence from a (possibly pre-aligned) *lsb_plane*.

        Parameters
        ----------
        lsb_plane : ndarray of shape (H, W), uint8
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
            lsb_plane, self.patch_size, self.message_length * self.n_repeats, self.key, self.tile_mode
        )


    # ---------------------------------------------------------------------------
    # Decode
    # ---------------------------------------------------------------------------

    def decode(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
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
        """
        img_u8 = img_as_ubyte(image)

        # Extract LSB plane (green channel for colour images)
        channel = img_u8[..., 1] if img_u8.ndim == 3 else img_u8
        lsb     = (channel & np.uint8(1)).astype(np.uint8)

        # -----------------------------------------------------------------------
        # Step 1 — direct decode (no alignment)
        # -----------------------------------------------------------------------
        bits, conf = extract_bits_from_lsb(lsb, self.patch_size, self.message_length * self.n_repeats, self.key, self.tile_mode)
        bits = np.mean(bits.reshape(self.n_repeats, -1), axis=0) > 0.5  # Majority vote over repeats
        if not self.sync_enabled or conf >= self.confidence_threshold:
            return bits

        best_bits, best_conf = bits, conf

        # -----------------------------------------------------------------------
        # Step 2 — rotation + scale + translation correction
        # -----------------------------------------------------------------------
        lsb_rs = synchronise(
            lsb,
            upsample_factor_rs=self.upsample_factor_rs,
            upsample_factor_t=self.upsample_factor_t,
            patch_size=self.patch_size,
            key=self.key,
            tile_mode=self.tile_mode,
        )

        bits_rs, conf_rs = extract_bits_from_lsb(
            lsb_rs, self.patch_size, self.message_length * self.n_repeats, self.key, self.tile_mode
        )
        bits_rs = np.mean(bits_rs.reshape(self.n_repeats, -1), axis=0) > 0.5  # Majority vote over repeats
        if conf_rs > best_conf:
            best_bits, best_conf = bits_rs, conf_rs
        if best_conf >= self.confidence_threshold:
            return best_bits

        return best_bits
