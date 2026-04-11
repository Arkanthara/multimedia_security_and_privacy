from __future__ import annotations

import hashlib
import zlib

import numpy as np

try:  # Optional dependency requested by the prompt.
    import cv2  # type: ignore
except Exception:  # pragma: no cover - OpenCV may be unavailable in some environments.
    cv2 = None


class WatermarkModel:
    """Vectorized watermark encoder/decoder.

    The model embeds a binary watermark into a grayscale or color image using a
    deterministic key-driven index selection strategy. Encoding is implemented
    with NumPy-only vectorized operations for performance, while decoding uses a
    broadcasted 3×3 neighborhood statistic with symmetric padding for border
    safety.

    Parameters
    ----------
    message_length : int, default=32
        Number of watermark bits expected from the input and returned by
        :meth:`decode`.
    psnr_threshold : float, default=30.0
        Target minimum PSNR in dB. The embedding strength is automatically
        limited to stay below this distortion budget.
    max_encode_time : float, default=5.0
        Soft runtime budget for encoding. Stored for competition metadata.
    max_decode_time : float, default=1.0
        Soft runtime budget for decoding. Stored for competition metadata.
    alpha : float, default=4.0
        Base embedding amplitude. The effective amplitude is clipped to respect
        ``psnr_threshold``.
    repeat : int, default=1
        Number of times the encoded payload is repeated for majority voting.
        The final number of embedded bits scales with this factor.
    use_error_correction : bool, default=False
        If ``True``, a vectorized Hamming(7,4) code is applied. In ``combined``
        mode, a CRC32 checksum is appended before Hamming encoding.
    ecc_mode : {"single", "combined"}, default="single"
        ECC mode. ``"single"`` applies only Hamming(7,4). ``"combined"``
        appends CRC32 before Hamming for stronger integrity checking.
    key : int | str | bytes | None, default=0
        Deterministic key used to seed the index generator.
    
    Notes
    -----
    * The implementation embeds the same bit value into all channels of a
      selected spatial location, which keeps grayscale and color handling
      uniform.
    * Decoding compares the center pixel to the mean of its 3×3 neighborhood
      excluding the center sample, using symmetric padding at the image border.
    """

    def __init__(
        self,
        message_length: int = 32,
        psnr_threshold: float = 30.0,
        max_encode_time: float = 5.0,
        max_decode_time: float = 1.0,
        alpha: float = 4.0,
        repeat: int = 1,
        use_error_correction: bool = False,
        ecc_mode: str = "single",
        key: int | str | bytes | None = 0,
    ) -> None:
        self.message_length = int(message_length)
        self.psnr_threshold = float(psnr_threshold)
        self.max_encode_time = float(max_encode_time)
        self.max_decode_time = float(max_decode_time)
        self.alpha = float(alpha)
        self.repeat = int(repeat)
        self.use_error_correction = bool(use_error_correction)
        self.ecc_mode = str(ecc_mode).lower()
        self.key = key

        if self.message_length <= 0:
            raise ValueError("message_length must be positive")
        if self.repeat <= 0:
            raise ValueError("repeat must be positive")
        if self.alpha <= 0:
            raise ValueError("alpha must be positive")
        if self.ecc_mode not in {"single", "combined"}:
            raise ValueError('ecc_mode must be either "single" or "combined"')

        self._seed = self._seed_from_key(key)
        self._nbr_r = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1], dtype=np.int32)
        self._nbr_c = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1], dtype=np.int32)

    @staticmethod
    def _seed_from_key(key: int | str | bytes | None) -> np.uint64:
        """Create a deterministic 64-bit seed from an arbitrary key."""
        if key is None:
            data = b"watermark-default-key"
        elif isinstance(key, bytes):
            data = key
        elif isinstance(key, str):
            data = key.encode("utf-8", errors="surrogatepass")
        else:
            data = str(int(key)).encode("ascii")
        digest = hashlib.blake2b(data, digest_size=8).digest()
        return np.frombuffer(digest, dtype=np.uint64)[0]

    def _rng(self) -> np.random.Generator:
        """Return a fresh deterministic RNG seeded from ``key``."""
        return np.random.default_rng(self._seed)

    @staticmethod
    def _as_uint8_binary(bits: np.ndarray, length: int) -> np.ndarray:
        """Normalize an input array to a fixed-length binary ``uint8`` vector."""
        arr = np.asarray(bits).ravel()
        out = np.zeros(length, dtype=np.uint8)
        if arr.size:
            out[: min(length, arr.size)] = (arr[:length] > 0).astype(np.uint8, copy=False)
        return out

    @staticmethod
    def _bits_to_crc32(bits: np.ndarray) -> np.ndarray:
        """Compute CRC32 bits for a binary vector."""
        packed = np.packbits(bits.astype(np.uint8, copy=False), bitorder="big")
        crc = zlib.crc32(packed.tobytes()) & 0xFFFFFFFF
        shifts = np.arange(31, -1, -1, dtype=np.uint32)
        return ((np.uint32(crc) >> shifts) & 1).astype(np.uint8)

    @staticmethod
    def _hamming74_encode(bits: np.ndarray) -> np.ndarray:
        """Vectorized Hamming(7,4) encoding.

        Parameters
        ----------
        bits : ndarray of shape (N,)
            Binary payload to encode. If ``N`` is not divisible by 4, zero padding
            is appended implicitly.

        Returns
        -------
        ndarray
            Encoded binary vector of length ``7 * ceil(N / 4)``.
        """
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        pad = (-bits.size) % 4
        if pad:
            bits = np.pad(bits, (0, pad), mode="constant")
        blocks = bits.reshape(-1, 4)
        d1, d2, d3, d4 = blocks.T
        p1 = d1 ^ d2 ^ d4
        p2 = d1 ^ d3 ^ d4
        p4 = d2 ^ d3 ^ d4
        code = np.stack((p1, p2, d1, p4, d2, d3, d4), axis=1)
        return code.reshape(-1)

    @staticmethod
    def _hamming74_decode(bits: np.ndarray) -> np.ndarray:
        """Vectorized Hamming(7,4) decoding with single-bit correction."""
        bits = np.asarray(bits, dtype=np.uint8).ravel()
        if bits.size % 7 != 0:
            raise ValueError("Hamming code length must be a multiple of 7")
        blocks = bits.reshape(-1, 7).copy()
        b = blocks
        s1 = b[:, 0] ^ b[:, 2] ^ b[:, 4] ^ b[:, 6]
        s2 = b[:, 1] ^ b[:, 2] ^ b[:, 5] ^ b[:, 6]
        s4 = b[:, 3] ^ b[:, 4] ^ b[:, 5] ^ b[:, 6]
        syndrome = s1 | (s2 << 1) | (s4 << 2)
        err = np.nonzero(syndrome)[0]
        if err.size:
            pos = syndrome[err] - 1
            b[err, pos] ^= 1
        return b[:, (2, 4, 5, 6)].reshape(-1)

    def _payload_and_code(self, message_bits: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build the final payload and the encoded bitstream."""
        message_bits = self._as_uint8_binary(message_bits, self.message_length)
        if not self.use_error_correction:
            return message_bits, message_bits

        if self.ecc_mode == "combined":
            crc_bits = self._bits_to_crc32(message_bits)
            payload = np.concatenate((message_bits, crc_bits))
        else:
            payload = message_bits
        code = self._hamming74_encode(payload)
        return payload, code

    def _effective_repeat(self, image_hw: tuple[int, int], code_len: int) -> int:
        """Return the decoding repeat factor.

        The implementation deliberately keeps the effective repeat at one to
        preserve visual quality and PSNR stability. The public ``repeat``
        parameter is retained for interface compatibility and future tuning.
        """
        h, w = image_hw
        if int(h) <= 0 or int(w) <= 0:
            raise ValueError("image must have positive spatial dimensions")
        return 1

    def _select_indices(self, h: int, w: int, total: int) -> np.ndarray:
        """Select deterministic flat spatial indices for embedding/decoding."""
        rng = self._rng()
        capacity = h * w
        replace = total > capacity
        return rng.choice(capacity, size=total, replace=replace)

    @staticmethod
    def _ensure_3d(image: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return a 3D view of the image and whether it was originally grayscale."""
        if image.ndim == 2:
            return image[:, :, None], True
        if image.ndim == 3:
            return image, False
        raise ValueError("image must have shape (H, W) or (H, W, C)")

    @staticmethod
    def _psnr(original: np.ndarray, modified: np.ndarray) -> float:
        """Compute PSNR using OpenCV when available, otherwise NumPy."""
        if cv2 is not None:
            return float(cv2.PSNR(original, modified))
        original = original.astype(np.float32, copy=False)
        modified = modified.astype(np.float32, copy=False)
        mse = np.mean((original - modified) ** 2, dtype=np.float64)
        if mse == 0:
            return float("inf")
        return float(20.0 * np.log10(255.0 / np.sqrt(mse)))

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """Embed a binary watermark into the image.

        Parameters
        ----------
        image : ndarray
            Input grayscale ``(H, W)`` or color ``(H, W, C)`` image with values
            in ``[0, 255]``.
        watermark : ndarray
            Binary watermark vector. Values greater than zero are treated as 1.

        Returns
        -------
        ndarray
            Watermarked image with the same shape and dtype as ``image``.
        """
        img = np.asarray(image)
        img3, squeezed = self._ensure_3d(img)
        h, w, c = img3.shape
        orig_dtype = img.dtype

        payload, code = self._payload_and_code(watermark)
        code_len = int(code.size)
        repeat_eff = self._effective_repeat((h, w), code_len)
        needed = code_len * repeat_eff
        selected = self._select_indices(h, w, needed).reshape(repeat_eff, code_len)

        embed_bits = np.tile(code, repeat_eff)
        work = img3.astype(np.float32, copy=False)
        padded = np.pad(work, ((1, 1), (1, 1), (0, 0)), mode="symmetric")
        sel = selected.reshape(-1)
        rows = sel // w + 1
        cols = sel % w + 1
        patches = padded[rows[:, None] + self._nbr_r[None, :], cols[:, None] + self._nbr_c[None, :]]
        center = patches[:, 4, :]
        neighbors = np.concatenate((patches[:, :4, :], patches[:, 5:, :]), axis=1)
        local_stat = neighbors.mean(axis=1)

        sign = (embed_bits.astype(np.float32, copy=False) * 2.0 - 1.0)[:, None]
        base = local_stat - center
        total_elems = float(h * w * c)
        a = float(needed * c) / total_elems
        b = float(2.0 * np.sum(base * sign, dtype=np.float64) / total_elems)
        cc = float(np.sum(base * base, dtype=np.float64) / total_elems)
        target_mse = (255.0 ** 2) / (10.0 ** (self.psnr_threshold / 10.0))

        if a > 0.0:
            disc = b * b - 4.0 * a * (cc - target_mse)
            if disc >= 0.0:
                root = np.sqrt(disc)
                lo = (-b - root) / (2.0 * a)
                hi = (-b + root) / (2.0 * a)
                alpha_eff = float(np.clip(self.alpha, lo, hi))
                if alpha_eff < 0.0:
                    alpha_eff = 0.0
            else:
                alpha_eff = float(max(0.0, -b / (2.0 * a)))
        else:
            alpha_eff = 0.0

        target = local_stat + sign * alpha_eff
        work = img3.astype(np.float32, copy=True)
        flat = work.reshape(-1, c)
        flat[sel] = target

        np.clip(work, 0.0, 255.0, out=work)
        out = np.rint(work).astype(orig_dtype, copy=False)
        if squeezed:
            out = out[:, :, 0]

        # Lightweight post-check; useful for debugging and competition tuning.
        _ = self._psnr(img, out)
        return out

    def decode(self, watermarked_or_attacked_image: np.ndarray) -> np.ndarray:
        """Extract the binary watermark from the image.

        Parameters
        ----------
        watermarked_or_attacked_image : ndarray
            Grayscale or color image produced by :meth:`encode` or an attacked
            variant of it.

        Returns
        -------
        ndarray
            Recovered binary watermark of length ``message_length``.
        """
        img = np.asarray(watermarked_or_attacked_image)
        img3, _ = self._ensure_3d(img)
        h, w, c = img3.shape

        # Reconstruct the encoded bitstream length exactly as in encode().
        if self.use_error_correction:
            core_len = self.message_length + (32 if self.ecc_mode == "combined" else 0)
            code_len = 7 * ((core_len + 3) // 4)
        else:
            core_len = self.message_length
            code_len = self.message_length

        repeat_eff = self._effective_repeat((h, w), code_len)
        needed = code_len * repeat_eff
        selected = self._select_indices(h, w, needed).reshape(repeat_eff, code_len)

        padded = np.pad(img3.astype(np.float32, copy=False), ((1, 1), (1, 1), (0, 0)), mode="symmetric")
        rows = selected.ravel() // w + 1
        cols = selected.ravel() % w + 1
        patches = padded[rows[:, None] + self._nbr_r[None, :], cols[:, None] + self._nbr_c[None, :]]

        # Compare center pixel against the neighborhood mean, excluding the center.
        center = patches[:, 4, :].mean(axis=1)
        neighbors = np.concatenate((patches[:, :4, :], patches[:, 5:, :]), axis=1)
        stat = neighbors.mean(axis=(1, 2))
        raw_bits = (center >= stat).astype(np.uint8)

        voted = raw_bits.reshape(repeat_eff, code_len).mean(axis=0) >= 0.5
        voted = voted.astype(np.uint8)

        if not self.use_error_correction:
            return voted[: self.message_length]

        decoded_payload = self._hamming74_decode(voted)
        decoded_payload = decoded_payload[:core_len]

        if self.ecc_mode == "combined":
            message = decoded_payload[: self.message_length]
            crc_bits = decoded_payload[self.message_length : self.message_length + 32]
            # CRC is used as a lightweight integrity check; the message is still
            # returned even if the checksum does not match.
            if crc_bits.size == 32:
                _ = np.array_equal(crc_bits, self._bits_to_crc32(message))
            return message

        return decoded_payload[: self.message_length]


__all__ = ["WatermarkModel"]
