"""High-performance vectorized watermarking model.

This module provides a competition-oriented watermarking system with deterministic
keyed embedding, repeated bit voting, optional ECC, and vectorized local
statistics decoding.
"""

from __future__ import annotations

import hashlib
import time
import zlib
from typing import Any

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - fallback when OpenCV is unavailable.
    cv2 = None


class WatermarkModel:
    """Vectorized additive watermarking model.

    Parameters
    ----------
    message_length : int, default=32
        Number of payload bits expected from the user watermark.
    psnr_threshold : float, default=30.0
        Minimum target PSNR used to upper-bound embedding strength.
    max_encode_time : float, default=5.0
        Soft time budget (seconds) for `encode`. Stored for benchmark reporting.
    max_decode_time : float, default=1.0
        Soft time budget (seconds) for `decode`. Stored for benchmark reporting.
    key : int | str, default=1337
        Secret key used to deterministically regenerate embedding indices.
    alpha : float, default=3.0
        Additive embedding amplitude before PSNR-based adjustment.
    repeat : int, default=5
        Number of repetitions of the encoded payload for majority voting.
    use_error_correction : bool, default=False
        Enable ECC processing before repetition.
    ecc_mode : {'hamming', 'crc', 'crc_hamming'}, default='hamming'
        ECC strategy when `use_error_correction=True`.
        - `hamming`: Hamming(7,4) single-error correction.
        - `crc`: CRC32 integrity bits.
        - `crc_hamming`: CRC32 followed by Hamming(7,4) (combined mode).
    local_statistic : {'mean', 'median'}, default='mean'
        Statistic used in decoding over 3x3 neighborhood excluding center.

    Notes
    -----
    The model handles grayscale `(H, W)` and color `(H, W, C)` images in [0, 255].
    Core embedding and decoding operations are fully vectorized with NumPy.
    """

    _ECC_MODES = {"hamming", "crc", "crc_hamming"}

    def __init__(
        self,
        message_length: int = 32,
        psnr_threshold: float = 30.0,
        max_encode_time: float = 5.0,
        max_decode_time: float = 1.0,
        *,
        key: int | str = 1337,
        alpha: float = 3.0,
        repeat: int = 5,
        use_error_correction: bool = False,
        ecc_mode: str = "hamming",
        local_statistic: str = "mean",
    ) -> None:
        if message_length <= 0:
            raise ValueError("message_length must be strictly positive.")
        if repeat <= 0:
            raise ValueError("repeat must be strictly positive.")
        if alpha <= 0:
            raise ValueError("alpha must be strictly positive.")

        local_statistic = local_statistic.lower()
        if local_statistic not in {"mean", "median"}:
            raise ValueError("local_statistic must be either 'mean' or 'median'.")

        self.message_length = int(message_length)
        self.psnr_threshold = float(psnr_threshold)
        self.max_encode_time = float(max_encode_time)
        self.max_decode_time = float(max_decode_time)
        self.alpha = float(alpha)
        self.repeat = int(repeat)
        self.local_statistic = local_statistic

        self.use_error_correction = bool(use_error_correction)
        if self.use_error_correction:
            ecc_mode = ecc_mode.lower()
            if ecc_mode not in self._ECC_MODES:
                allowed = ", ".join(sorted(self._ECC_MODES))
                raise ValueError(f"ecc_mode must be one of: {allowed}.")
            self.ecc_mode = ecc_mode
        else:
            self.ecc_mode = "none"

        self._seed = self._stable_seed(key)
        self._encoded_length = self._compute_encoded_length()
        self._total_bits = self._encoded_length * self.repeat

        self._eligible_cache: dict[tuple[int, int, int], np.ndarray] = {}
        self._index_cache: dict[tuple[tuple[int, int, int], int], np.ndarray] = {}

        self.last_encode_time: float | None = None
        self.last_decode_time: float | None = None
        self.last_psnr: float | None = None
        self.last_crc_ok: bool | None = None

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """Embed a binary watermark into an image.

        Parameters
        ----------
        image : numpy.ndarray
            Input image of shape `(H, W)` or `(H, W, C)` with values in [0, 255].
        watermark : numpy.ndarray
            Binary watermark payload. It is flattened; only the first
            `message_length` bits are used, and missing bits are zero-padded.

        Returns
        -------
        numpy.ndarray
            Watermarked image with the same shape as `image` and dtype `uint8`.

        Raises
        ------
        ValueError
            If the image cannot host the requested payload size.
        """
        t0 = time.perf_counter()

        image_u8 = self._to_uint8_image(image)
        image_f32, squeeze_channel = self._as_3d_float(image_u8)

        message_bits = self._prepare_message_bits(watermark)
        encoded_bits = self._encode_ecc(message_bits)
        repeated_bits = np.tile(encoded_bits, self.repeat)

        indices = self._get_indices(image_f32.shape, repeated_bits.size)
        alpha = self._effective_alpha(image_f32.shape, repeated_bits.size)

        flat = image_f32.reshape(-1)
        signed = repeated_bits.astype(np.float32) * 2.0 - 1.0
        flat[indices] = np.clip(flat[indices] + signed * alpha, 0.0, 255.0)

        watermarked_3d = np.rint(flat.reshape(image_f32.shape)).astype(np.uint8)
        watermarked = watermarked_3d[..., 0] if squeeze_channel else watermarked_3d

        self.last_psnr = self._psnr(image_u8, watermarked)
        self.last_encode_time = time.perf_counter() - t0
        return watermarked

    def decode(self, watermarked_or_attacked_image: np.ndarray) -> np.ndarray:
        """Extract the binary watermark from an image.

        Parameters
        ----------
        watermarked_or_attacked_image : numpy.ndarray
            Watermarked or attacked image of shape `(H, W)` or `(H, W, C)`.

        Returns
        -------
        numpy.ndarray
            Decoded binary watermark as a 1D `uint8` array of length
            `message_length`.

        Raises
        ------
        ValueError
            If the image cannot host the configured payload size.
        """
        t0 = time.perf_counter()

        image_u8 = self._to_uint8_image(watermarked_or_attacked_image)
        image_f32, _ = self._as_3d_float(image_u8)

        indices = self._get_indices(image_f32.shape, self._total_bits)
        raw_bits = self._decode_bits_from_indices(image_f32, indices)

        if self.repeat == 1:
            voted = raw_bits
        else:
            vote_sum = raw_bits.reshape(self.repeat, self._encoded_length).sum(axis=0)
            voted = (vote_sum * 2 >= self.repeat).astype(np.uint8)

        decoded = self._decode_ecc(voted)

        self.last_decode_time = time.perf_counter() - t0
        return decoded

    def _compute_encoded_length(self) -> int:
        """Compute encoded payload length before repetition.

        Returns
        -------
        int
            Number of bits embedded per repetition.
        """
        if not self.use_error_correction:
            return self.message_length
        if self.ecc_mode == "crc":
            return self.message_length + 32
        if self.ecc_mode == "hamming":
            return ((self.message_length + 3) // 4) * 7
        if self.ecc_mode == "crc_hamming":
            return (((self.message_length + 32) + 3) // 4) * 7
        raise RuntimeError("Unexpected ECC mode.")

    @staticmethod
    def _stable_seed(key: Any) -> int:
        """Build a reproducible 64-bit seed from an integer or string key.

        Parameters
        ----------
        key : Any
            User-provided key.

        Returns
        -------
        int
            Deterministic 64-bit seed.
        """
        if isinstance(key, (int, np.integer)):
            return int(int(key) % (1 << 64))
        digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, byteorder="little", signed=False)

    def _seed_for_shape(self, shape: tuple[int, int, int], total_bits: int) -> int:
        """Combine base key seed with image-dependent constants.

        Parameters
        ----------
        shape : tuple of int
            Image shape as `(H, W, C)`.
        total_bits : int
            Number of embedding indices needed.

        Returns
        -------
        int
            Deterministic seed for index generation.
        """
        h, w, c = shape
        mix = (
            np.uint64(h) * np.uint64(73856093)
            ^ np.uint64(w) * np.uint64(19349663)
            ^ np.uint64(c) * np.uint64(83492791)
            ^ np.uint64(total_bits) * np.uint64(2654435761)
            ^ np.uint64(self._encoded_length) * np.uint64(2246822519)
        )
        return int(np.uint64(self._seed) ^ mix)

    def _to_uint8_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize input image into `uint8` array.

        Parameters
        ----------
        image : numpy.ndarray
            Input image array.

        Returns
        -------
        numpy.ndarray
            Clipped `uint8` image.
        """
        arr = np.asarray(image)
        if arr.ndim not in (2, 3):
            raise ValueError("image must have shape (H, W) or (H, W, C).")
        if arr.dtype == np.uint8:
            return arr.copy()
        return np.clip(np.rint(arr), 0, 255).astype(np.uint8)

    @staticmethod
    def _as_3d_float(image_u8: np.ndarray) -> tuple[np.ndarray, bool]:
        """Convert image to float32 `(H, W, C)` representation.

        Parameters
        ----------
        image_u8 : numpy.ndarray
            Input `uint8` image with 2D or 3D shape.

        Returns
        -------
        image_3d : numpy.ndarray
            Float32 image of shape `(H, W, C)`.
        squeeze_channel : bool
            Whether the input was grayscale and should be squeezed on output.
        """
        if image_u8.ndim == 2:
            return image_u8[..., None].astype(np.float32), True
        return image_u8.astype(np.float32), False

    def _prepare_message_bits(self, watermark: np.ndarray) -> np.ndarray:
        """Create fixed-length binary message vector.

        Parameters
        ----------
        watermark : numpy.ndarray
            User watermark bits.

        Returns
        -------
        numpy.ndarray
            Binary vector of length `message_length` and dtype `uint8`.
        """
        bits = (np.asarray(watermark).reshape(-1) > 0).astype(np.uint8)
        if bits.size >= self.message_length:
            return bits[: self.message_length]
        out = np.zeros(self.message_length, dtype=np.uint8)
        out[: bits.size] = bits
        return out

    def _get_eligible_indices(self, shape: tuple[int, int, int]) -> np.ndarray:
        """Return flattened pixel indices eligible for embedding.

        Parameters
        ----------
        shape : tuple of int
            Image shape `(H, W, C)`.

        Returns
        -------
        numpy.ndarray
            1D array of flattened indices over all channels.

        Notes
        -----
        When possible, only interior pixels are used for stronger local-stat
        decoding and cleaner neighborhood extraction.
        """
        cached = self._eligible_cache.get(shape)
        if cached is not None:
            return cached

        h, w, c = shape
        if h >= 3 and w >= 3:
            rows = np.arange(1, h - 1, dtype=np.int64)
            cols = np.arange(1, w - 1, dtype=np.int64)
            base = ((rows[:, None] * w) + cols[None, :]).reshape(-1) * c
        else:
            base = np.arange(h * w, dtype=np.int64) * c

        if c == 1:
            eligible = base
        else:
            eligible = (base[:, None] + np.arange(c, dtype=np.int64)[None, :]).reshape(-1)

        self._eligible_cache[shape] = eligible
        return eligible

    def _get_indices(self, shape: tuple[int, int, int], total_bits: int) -> np.ndarray:
        """Generate deterministic embedding indices for a given shape.

        Parameters
        ----------
        shape : tuple of int
            Image shape `(H, W, C)`.
        total_bits : int
            Number of required index positions.

        Returns
        -------
        numpy.ndarray
            Flattened index array of length `total_bits`.

        Raises
        ------
        ValueError
            If `total_bits` exceeds image capacity.
        """
        cache_key = (shape, total_bits)
        cached = self._index_cache.get(cache_key)
        if cached is not None:
            return cached

        eligible = self._get_eligible_indices(shape)
        if total_bits > eligible.size:
            raise ValueError(
                "Payload too large for image capacity. "
                f"Required {total_bits} bits, capacity {eligible.size}."
            )

        rng = np.random.default_rng(self._seed_for_shape(shape, total_bits))
        chosen = rng.choice(eligible.size, size=total_bits, replace=False)
        indices = eligible[chosen]

        self._index_cache[cache_key] = indices
        return indices

    def _effective_alpha(self, shape: tuple[int, int, int], total_bits: int) -> float:
        """Compute PSNR-safe embedding amplitude.

        Parameters
        ----------
        shape : tuple of int
            Image shape `(H, W, C)`.
        total_bits : int
            Number of modified pixels/channels.

        Returns
        -------
        float
            Effective embedding amplitude.
        """
        if self.psnr_threshold <= 0:
            return self.alpha
        n_values = float(np.prod(shape, dtype=np.int64))
        max_mse = (255.0 * 255.0) / (10.0 ** (self.psnr_threshold / 10.0))
        alpha_upper = np.sqrt(max_mse * n_values / float(total_bits))
        return float(min(self.alpha, 0.98 * alpha_upper))

    def _decode_bits_from_indices(self, image_3d: np.ndarray, indices: np.ndarray) -> np.ndarray:
        """Decode raw bits from selected indices using local 3x3 context.

        Parameters
        ----------
        image_3d : numpy.ndarray
            Float32 image of shape `(H, W, C)`.
        indices : numpy.ndarray
            Flattened index array.

        Returns
        -------
        numpy.ndarray
            Raw decoded bits as a 1D `uint8` array.
        """
        h, w, c_count = image_3d.shape

        pix = indices // c_count
        ch = indices % c_count
        y = pix // w
        x = pix % w

        pad = np.pad(image_3d, ((1, 1), (1, 1), (0, 0)), mode="symmetric")
        yp = y + 1
        xp = x + 1

        n0 = pad[yp - 1, xp - 1, ch]
        n1 = pad[yp - 1, xp, ch]
        n2 = pad[yp - 1, xp + 1, ch]
        n3 = pad[yp, xp - 1, ch]
        n4 = pad[yp, xp + 1, ch]
        n5 = pad[yp + 1, xp - 1, ch]
        n6 = pad[yp + 1, xp, ch]
        n7 = pad[yp + 1, xp + 1, ch]

        if self.local_statistic == "mean":
            local_stat = (n0 + n1 + n2 + n3 + n4 + n5 + n6 + n7) * 0.125
        else:
            neighbors = np.stack((n0, n1, n2, n3, n4, n5, n6, n7), axis=1)
            local_stat = np.median(neighbors, axis=1)

        center = image_3d[y, x, ch]
        return (center >= local_stat).astype(np.uint8)

    @staticmethod
    def _bits_to_bytes(bits: np.ndarray) -> bytes:
        """Pack binary vector into bytes with big-endian bit order.

        Parameters
        ----------
        bits : numpy.ndarray
            1D binary vector.

        Returns
        -------
        bytes
            Packed bytes representation.
        """
        if bits.size == 0:
            return b""
        pad = (-bits.size) % 8
        if pad:
            bits = np.pad(bits, (0, pad))
        return np.packbits(bits, bitorder="big").tobytes()

    @staticmethod
    def _crc32_bits(bits: np.ndarray) -> np.ndarray:
        """Compute CRC32 and return it as 32 binary bits.

        Parameters
        ----------
        bits : numpy.ndarray
            Input binary vector.

        Returns
        -------
        numpy.ndarray
            32-bit CRC as `uint8` vector.
        """
        crc = zlib.crc32(WatermarkModel._bits_to_bytes(bits)) & 0xFFFFFFFF
        crc_bytes = np.array([crc], dtype=">u4").view(np.uint8)
        return np.unpackbits(crc_bytes, bitorder="big").astype(np.uint8)

    @staticmethod
    def _hamming74_encode(bits: np.ndarray) -> np.ndarray:
        """Vectorized Hamming(7,4) encoder.

        Parameters
        ----------
        bits : numpy.ndarray
            Input message bits.

        Returns
        -------
        numpy.ndarray
            Encoded bitstream.
        """
        pad = (-bits.size) % 4
        if pad:
            bits = np.pad(bits, (0, pad))
        d = bits.reshape(-1, 4)
        p1 = d[:, 0] ^ d[:, 1] ^ d[:, 3]
        p2 = d[:, 0] ^ d[:, 2] ^ d[:, 3]
        p3 = d[:, 1] ^ d[:, 2] ^ d[:, 3]
        code = np.stack((p1, p2, d[:, 0], p3, d[:, 1], d[:, 2], d[:, 3]), axis=1)
        return code.reshape(-1).astype(np.uint8)

    @staticmethod
    def _hamming74_decode(bits: np.ndarray, out_len: int) -> np.ndarray:
        """Vectorized Hamming(7,4) decoder with single-bit correction.

        Parameters
        ----------
        bits : numpy.ndarray
            Hamming-coded bitstream.
        out_len : int
            Number of decoded bits to return after depadding.

        Returns
        -------
        numpy.ndarray
            Decoded binary vector of length `out_len`.
        """
        if bits.size == 0:
            return np.zeros(out_len, dtype=np.uint8)

        if bits.size % 7:
            bits = bits[: bits.size - (bits.size % 7)]

        cw = bits.reshape(-1, 7).astype(np.uint8)

        s1 = cw[:, 0] ^ cw[:, 2] ^ cw[:, 4] ^ cw[:, 6]
        s2 = cw[:, 1] ^ cw[:, 2] ^ cw[:, 5] ^ cw[:, 6]
        s3 = cw[:, 3] ^ cw[:, 4] ^ cw[:, 5] ^ cw[:, 6]

        syndrome = s1 + (s2 << 1) + (s3 << 2)
        err_rows = np.flatnonzero(syndrome)
        if err_rows.size:
            err_cols = syndrome[err_rows] - 1
            cw[err_rows, err_cols] ^= 1

        decoded = cw[:, (2, 4, 5, 6)].reshape(-1)
        if decoded.size < out_len:
            out = np.zeros(out_len, dtype=np.uint8)
            out[: decoded.size] = decoded
            return out
        return decoded[:out_len].astype(np.uint8)

    def _encode_ecc(self, message_bits: np.ndarray) -> np.ndarray:
        """Apply ECC pipeline to message bits.

        Parameters
        ----------
        message_bits : numpy.ndarray
            Message bit vector of length `message_length`.

        Returns
        -------
        numpy.ndarray
            ECC-encoded payload bits.
        """
        if not self.use_error_correction:
            return message_bits

        if self.ecc_mode == "crc":
            return np.concatenate((message_bits, self._crc32_bits(message_bits)))
        if self.ecc_mode == "hamming":
            return self._hamming74_encode(message_bits)
        if self.ecc_mode == "crc_hamming":
            base = np.concatenate((message_bits, self._crc32_bits(message_bits)))
            return self._hamming74_encode(base)
        raise RuntimeError("Unexpected ECC mode.")

    def _decode_ecc(self, coded_bits: np.ndarray) -> np.ndarray:
        """Decode ECC payload and return fixed-length message bits.

        Parameters
        ----------
        coded_bits : numpy.ndarray
            Majority-voted encoded payload bits.

        Returns
        -------
        numpy.ndarray
            Decoded watermark bits of length `message_length`.
        """
        self.last_crc_ok = None

        if not self.use_error_correction:
            return coded_bits[: self.message_length].astype(np.uint8)

        if self.ecc_mode == "crc":
            msg = coded_bits[: self.message_length].astype(np.uint8)
            recv_crc = coded_bits[self.message_length : self.message_length + 32].astype(np.uint8)
            self.last_crc_ok = bool(np.array_equal(recv_crc, self._crc32_bits(msg)))
            return msg

        if self.ecc_mode == "hamming":
            return self._hamming74_decode(coded_bits, self.message_length)

        if self.ecc_mode == "crc_hamming":
            dec = self._hamming74_decode(coded_bits, self.message_length + 32)
            msg = dec[: self.message_length]
            recv_crc = dec[self.message_length : self.message_length + 32]
            self.last_crc_ok = bool(np.array_equal(recv_crc, self._crc32_bits(msg)))
            return msg.astype(np.uint8)

        raise RuntimeError("Unexpected ECC mode.")

    @staticmethod
    def _psnr(reference: np.ndarray, compared: np.ndarray) -> float:
        """Compute PSNR between two images.

        Parameters
        ----------
        reference : numpy.ndarray
            Original image.
        compared : numpy.ndarray
            Distorted image.

        Returns
        -------
        float
            PSNR value in dB.
        """
        if cv2 is not None:
            return float(cv2.PSNR(reference, compared))
        ref = reference.astype(np.float32)
        cmp = compared.astype(np.float32)
        mse = np.mean((ref - cmp) ** 2)
        if mse <= 0.0:
            return float("inf")
        return float(10.0 * np.log10((255.0 * 255.0) / mse))
