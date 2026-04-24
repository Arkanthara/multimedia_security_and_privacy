"""
spatial/model.py
================
Spatial-domain watermark — encode and decode entry points.

Encode pipeline
---------------
    image → preprocess watermark (ECC + repetition)
          → build bipolar patch  (size: patch_size × upsample_factor)
          → compute NVF
          → img = img + (alpha_1 * NVF + alpha_2 * (1 - NVF)) * w

Decode pipeline
---------------
    image → compute NVF
          → Wiener denoising (noise power estimated from weighted reference)
          → residual = image − denoised
          → synchronise residual
              → returns aligned summed block of shape
                (patch_size * upsample_factor, patch_size * upsample_factor)
          → extract bits from the aligned block
          → post-process (majority vote + LDPC BP decode)

Key parameters
--------------
``upsample_factor``
    Controls how many image pixels represent one logical patch cell.
    The classic value is 2 (each patch cell → 2×2 pixels); setting it to 3
    gives denser embedding (3×3 pixels per cell).  **Must be identical at
    encode and decode time.**

``synchronise`` signature (utils/sync.py)
-----------------------------------------
    synchronise(
        lsb_image,           # residual or LSB plane, shape (H, W)
        patch_size=32,
        key=0,
        tile_mode="symmetric",
        nms_size_ac=31,      # NMS window for autocorrelation peaks
        nms_size_corr=5,     # NMS window for translation correlation
        upsample_factor=2,
    ) → aligned_block of shape (patch_size * upsample_factor,
                                patch_size * upsample_factor)

    The returned block is already accumulated (summed over all tiles and
    translation-corrected).  Pass it directly to
    ``extract_bits_from_spatial``, which then performs only the
    upsample_factor×upsample_factor sub-block sum and bit sampling.
"""

from __future__ import annotations

import cv2
from matplotlib import image
import numpy as np
from scipy.ndimage import uniform_filter
from scipy.signal import wiener
from skimage.util import img_as_float, img_as_ubyte

from utils.error_correction import repetition_decode_ldpc, repetition_encode
from utils.patch import (
    build_patch_image_bipolar,
    build_reference_patch_bipolar,
    extract_bits_from_spatial,
)
from utils.sync import synchronise


class WatermarkModel:
    """
    Spatial-domain watermarking model using the Noise Visibility Function (NVF).

    The watermark is embedded as an additive signal weighted by the NVF:
    ``img_w = img + (alpha_1 * NVF + alpha_2 * (1 - NVF)) * w``

    where ``w`` is a bipolar ({-1, +1}) pseudo-random patch carrying the
    message.

    Parameters
    ----------
    alpha_1 : float
        Embedding strength in *high*-variance (textured) regions.
    alpha_2 : float
        Embedding strength in *low*-variance (smooth) regions.
    D : float
        NVF sharpness parameter.  Larger values reduce alpha_1's influence.
    use_nvf : bool
        If ``False``, embed uniformly: ``img_w = img + alpha_1 * w``.
    nvf_window_size : int
        Side length of the square window used to estimate local variance.
    msg_length : int
        Number of message bits.
    use_ecc : bool
        Apply repetition-code LDPC error correction before embedding.
    ecc_repetitions : int
        Number of repetitions per bit for the repetition code.
    msg_repetitions : int
        Number of times the (ECC-encoded) message is tiled across the patch.
    patch_size : int
        Side length of the square base patch (logical pixels, before
        upsampling).
    upsample_factor : int
        Number of image pixels per logical patch cell (default 2).
        Increasing this value gives each embedded bit a larger footprint,
        which can improve robustness at the cost of embedding capacity.
        Must be ≥ 1 and identical at encode and decode time.
    key : int
        Seed controlling all random operations (patch, bit positions).
    tile_mode : {"normal", "symmetric"}
        Patch tiling strategy — must match between encode and decode.
    """

    def __init__(
        self,
        alpha_1: float = 3.8,
        alpha_2: float = 10,
        D: float = 50.0,
        use_nvf: bool = True,
        nvf_window_size: int = 7,
        msg_length: int = 32,
        use_ecc: bool = False,
        ecc_repetitions: int = 3,
        msg_repetitions: int = 1,
        patch_size: int = 32,
        upsample_factor: int = 2,
        key: int = 42,
        tile_mode: str = "symmetric",
    ) -> None:
        self.alpha_1        = alpha_1 / 255.0  # Scale to [0, 1] range for float images
        self.alpha_2        = alpha_2 / 255.0
        self.D              = D / 255.0
        self.use_nvf        = use_nvf
        self.nvf_window_size = nvf_window_size
        self.msg_length     = msg_length
        self.use_ecc        = use_ecc
        self.ecc_repetitions = ecc_repetitions
        self.msg_repetitions = msg_repetitions
        self.patch_size     = patch_size
        self.upsample_factor = upsample_factor
        self.key            = key
        self.tile_mode      = tile_mode

    # -----------------------------------------------------------------------
    # Private helpers
    # -----------------------------------------------------------------------

    def _get_channel(self, img_f: np.ndarray) -> np.ndarray:
        """Return the embedding channel: green (index 1) for RGB, full for grayscale."""
        return img_f[..., 1] if img_f.ndim == 3 else img_f

    def _set_channel(self, img_f: np.ndarray, channel: np.ndarray) -> np.ndarray:
        """Write *channel* back into a copy of *img_f*."""
        if img_f.ndim == 3:
            out = img_f.copy()
            out[..., 1] = channel
            return out
        return channel

    def _compute_local_stats(self, image: np.ndarray, window_size: int) -> np.ndarray:
        kernel = np.ones((window_size , window_size), np.float32) / (window_size** 2)
        mean = cv2.filter2D(image , -1, kernel)
        variance = cv2.filter2D(image ** 2, -1, kernel) - mean ** 2
        return mean, variance

    def _wiener_filter(self, image: np.ndarray, window_size: int, noise_power: float) -> np.ndarray:
        local_mean , local_variance = self._compute_local_stats(image, window_size)
        wiener = local_mean + (local_variance / (local_variance + noise_power)) * (
        image - local_mean)
        return wiener

    def _compute_nvf(self, image: np.ndarray) -> np.ndarray:
        """
        Compute the Noise Visibility Function (NVF).

        ``NVF = 1 / (1 + D * local_var / max_var)``

        * High-variance (textured) regions → NVF ≈ 0
        * Smooth regions                   → NVF ≈ 1
        """
        _, local_var = self._compute_local_stats(image, self.nvf_window_size)
        max_var   = np.max(local_var)
        if max_var == 0.0:
            return np.ones_like(image, dtype=np.float32)
        return (1.0 / (1.0 + self.D * local_var / max_var)).astype(np.float32)

    def _embedding_strength(self, nvf: np.ndarray) -> np.ndarray:
        """Per-pixel embedding strength: ``alpha_1 * NVF + alpha_2 * (1 - NVF)``."""
        if self.use_nvf:
            return (self.alpha_1 * nvf + self.alpha_2 * (1.0 - nvf)).astype(np.float32)
        return np.full_like(nvf, self.alpha_1, dtype=np.float32)

    def _n_embedded_bits(self) -> int:
        """Total number of bits embedded in the patch (after ECC + repetition)."""
        n = self.msg_length
        if self.use_ecc:
            n *= self.ecc_repetitions
        return n * self.msg_repetitions

    def _preprocess_watermark(self, watermark: np.ndarray) -> np.ndarray:
        """
        Prepare the watermark for embedding.

        1. Encode with repetition ECC via the LDPC generator matrix.
        2. Tile the encoded message *msg_repetitions* times.

        Parameters
        ----------
        watermark : ndarray of shape (msg_length,), dtype uint8, values {0,1}

        Returns
        -------
        processed : ndarray of shape (_n_embedded_bits(),), dtype uint8
        """
        wm = watermark.astype(np.uint8)
        if self.use_ecc:
            wm = repetition_encode(wm, self.ecc_repetitions)
        if self.msg_repetitions > 1:
            wm = np.tile(wm, self.msg_repetitions)
        return wm

    def _postprocess_watermark(self, bits: np.ndarray) -> np.ndarray:
        """
        Recover the original message from extracted (noisy) bits.

        1. Majority vote over *msg_repetitions* copies.
        2. LDPC BP decode over ECC repetitions.

        Parameters
        ----------
        bits : ndarray of shape (_n_embedded_bits(),), dtype uint8

        Returns
        -------
        message : ndarray of shape (msg_length,), dtype uint8
        """
        if self.msg_repetitions > 1:
            n    = len(bits) // self.msg_repetitions
            bits = (
                bits[: n * self.msg_repetitions]
                .reshape(self.msg_repetitions, n)
                .mean(axis=0)
                > 0.5
            ).astype(np.uint8)
        if self.use_ecc:
            bits = repetition_decode_ldpc(bits, self.ecc_repetitions)
        return bits

    # -----------------------------------------------------------------------
    # Encode
    # -----------------------------------------------------------------------

    def encode(
        self,
        image: np.ndarray,
        watermark: np.ndarray,
    ) -> np.ndarray:
        """
        Embed *watermark* bits into *image* using spatial-domain NVF weighting.

        Parameters
        ----------
        image : ndarray of shape (H, W) or (H, W, C), any dtype
        watermark : ndarray of shape (msg_length,), dtype int, values {0, 1}

        Returns
        -------
        watermarked : ndarray, same shape as *image*, dtype uint8
        """
        img_f = img_as_float(image).astype(np.float32)
        H, W  = img_f.shape[:2]

        # 1. Preprocess watermark: ECC encode (via LDPC G matrix) + tile
        wm = self._preprocess_watermark(np.asarray(watermark, dtype=np.uint8))

        # 2. Build bipolar ({-1, +1}) patch of image size
        #    Each logical patch cell occupies upsample_factor × upsample_factor pixels
        w = build_patch_image_bipolar(
            wm, (H, W), self.patch_size, self.key, self.tile_mode,
            upsample_factor=self.upsample_factor,
        )

        # 3. Compute per-pixel embedding strength from the target channel
        channel  = self._get_channel(img_f)
        nvf      = self._compute_nvf(channel)
        strength = self._embedding_strength(nvf)

        # 4. Additive embedding
        embedded = np.clip(channel + strength * w, 0.0, 1.0)

        return img_as_ubyte(self._set_channel(img_f, embedded))

    # -----------------------------------------------------------------------
    # Decode
    # -----------------------------------------------------------------------

    def decode(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """
        Recover watermark bits from a (possibly transformed) watermarked *image*.

        Decode pipeline
        ---------------
        1. Extract the embedding channel (green for RGB, full for grayscale).
        2. Build the bipolar reference patch and compute NVF-weighted noise power.
        3. Wiener-filter the channel to suppress non-watermark content.
        4. Compute residual = channel − denoised.
        5. Call :func:`~utils.sync.synchronise` with the correct parameters
           to affine-correct and translation-align the residual, yielding an
           accumulated block of shape
           ``(patch_size * upsample_factor, patch_size * upsample_factor)``.
        6. Extract bits from the aligned block via
           :func:`~utils.patch.extract_bits_from_spatial`.
        7. Post-process: majority vote + LDPC BP decode.

        Parameters
        ----------
        image : ndarray of shape (H, W) or (H, W, C), any dtype

        Returns
        -------
        bits : ndarray of shape (msg_length,), dtype uint8, values {0, 1}
        """
        img_f = img_as_float(image).astype(np.float32)
        H, W  = img_f.shape[:2]

        channel = self._get_channel(img_f)   # (H, W)

        # 1. Bipolar reference patch (base pattern without watermark bits)
        ref = build_reference_patch_bipolar(
            (H, W), self.patch_size, self.key, self.tile_mode,
            upsample_factor=self.upsample_factor,
        )

        # 2. NVF-weighted reference models the embedded watermark signal
        nvf          = self._compute_nvf(channel)
        strength     = self._embedding_strength(nvf)
        ref_weighted = strength * ref

        # 3. Wiener denoising — noise power ≈ mean energy of the watermark signal
        noise_power = np.var(ref_weighted)
        denoised    = self._wiener_filter(channel, window_size=3, noise_power=noise_power)

        # 4. Residual ≈ watermark signal (float32, shape (H, W))
        residual = (channel - denoised).astype(np.float32)

        # 5. Synchronise residual with the reference.
        #
        #    synchronise() signature (utils/sync.py):
        #        synchronise(
        #            lsb_image,           ← residual (float32 OK)
        #            patch_size=...,
        #            key=...,
        #            tile_mode=...,
        #            nms_size_ac=31,      ← NMS window, autocorrelation peaks
        #            nms_size_corr=5,     ← NMS window, translation correlation
        #            upsample_factor=..., ← must match encode
        #        )
        #
        #    Returns: aligned_block of shape
        #        (patch_size * upsample_factor, patch_size * upsample_factor)
        #    — already affine-corrected, tiled, summed and translation-aligned.
        #    Do NOT pass the full-size residual to extract_bits_from_spatial;
        #    pass the aligned_block returned here.
        aligned_block = synchronise(
            residual,
            patch_size=self.patch_size,
            key=self.key,
            tile_mode=self.tile_mode,
            nms_size_ac=31,
            nms_size_corr=5,
            upsample_factor=self.upsample_factor,
        )

        # 6. Extract bits from the aligned accumulated block.
        #    Because aligned_block has shape (up, up) with up = patch_size *
        #    upsample_factor, extract_bits_from_spatial sees it as one tile and
        #    performs only the upsample_factor×upsample_factor sub-block sum
        #    plus keyed bit sampling — no further tiling accumulation occurs.
        n_bits = self._n_embedded_bits()
        bits, _ = extract_bits_from_spatial(
            aligned_block,
            self.patch_size,
            n_bits,
            self.key,
            self.tile_mode,
            upsample_factor=self.upsample_factor,
        )

        # 7. Post-process: majority vote + LDPC BP decode
        return self._postprocess_watermark(bits)
