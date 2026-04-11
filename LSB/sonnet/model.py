"""
model.py
========
Highly optimized, fully vectorized blind watermarking system.

Architecture
------------
Encoding
    Each bit shifts one selected pixel by ±alpha::

        bit = 1  →  pixel += alpha
        bit = 0  →  pixel -= alpha

Decoding
    For every embedded position the center pixel is compared to the mean
    of its configurable k×k neighbourhood (center excluded, same channel).
    Reflect-padding handles border pixels transparently::

        center ≥ local_mean  →  1
        center <  local_mean  →  0

Bit layout
----------
ECC disabled::

    payload = tile(message, repeat)               # (message_length × repeat,)
    → per-bit majority vote at decode time

ECC enabled (BCH)::

    payload = concat( message×1 ,  tile(ecc, repeat) )
    #                  ^once           ^repeat times

    The message is embedded exactly once.
    BCH parity bits are tiled *repeat* times, majority-voted at decode
    time, then fed to the BCH decoder to correct errors in the single
    embedded message copy.
    ``'combined'`` mode prepends CRC-32 of the message to the BCH-protected
    data for a post-correction integrity check.

Dependencies
------------
Required : numpy, opencv-python
Optional : bchlib  (ECC; falls back to pure repetition if absent)
"""

from __future__ import annotations

import binascii
from typing import Tuple

import cv2          # noqa: F401 – available for caller use (e.g. PSNR checks)
import numpy as np

try:
    import bchlib
    _BCH_AVAILABLE = True
except ImportError:
    _BCH_AVAILABLE = False


class WatermarkModel:
    """
    Vectorized blind watermarking model using local pixel modulation.

    Parameters
    ----------
    message_length : int, default 32
        Number of bits in the watermark message.
    psnr_threshold : float, default 30.0
        Minimum acceptable PSNR (dB). Informational only.
    max_encode_time : float, default 5.0
        Encoding time budget in seconds. Informational only.
    max_decode_time : float, default 1.0
        Decoding time budget in seconds. Informational only.
    key : int, default 42
        Seed for the deterministic PRNG (``numpy.random.default_rng``).
        Must be identical at encode and decode time.
    alpha : float, default 6.0
        Embedding strength. Larger → more robust, lower PSNR.
    repeat : int, default 5
        ECC disabled : message tiled *repeat* times; majority vote per bit.
        ECC enabled  : BCH parity tiled *repeat* times; message embedded once.
    use_error_correction : bool, default False
        Enable BCH error correction (requires ``bchlib``).
        Falls back silently to pure repetition if the library is absent.
    ecc_mode : str, default ``'single'``
        ``'single'``   — BCH parity protection only.
        ``'combined'`` — CRC-32 appended before BCH for integrity check.
    neighborhood_size : int, default 3
        Side length of the square neighbourhood used at decode time.
        Must be an odd integer ≥ 3 (e.g. 3 for 3×3, 5 for 5×5).
        A larger window is more robust to noise but slower.
    """

    def __init__(
        self,
        message_length: int    = 32,
        psnr_threshold: float  = 30.0,
        max_encode_time: float = 5.0,
        max_decode_time: float = 1.0,
        key: int               = 42,
        alpha: float           = 10.0,
        repeat: int            = 5,
        use_error_correction: bool = True,
        ecc_mode: str          = "combined",
        neighborhood_size: int = 3,
    ) -> None:
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")

        self.message_length       = message_length
        self.psnr_threshold       = psnr_threshold
        self.max_encode_time      = max_encode_time
        self.max_decode_time      = max_decode_time
        self.key                  = key
        self.alpha                = float(alpha)
        self.repeat               = repeat
        self.use_error_correction = use_error_correction
        self.ecc_mode             = ecc_mode
        self.neighborhood_size    = neighborhood_size

        self._setup_ecc()
        self._build_neighbor_offsets()

    # ------------------------------------------------------------------ #
    #  One-time setup                                                       #
    # ------------------------------------------------------------------ #

    def _setup_ecc(self) -> None:
        """
        Initialise the BCH codec and precompute bit-layout constants.

        Attributes set
        --------------
        _bch : bchlib.BCH or None
        _msg_byte_len : int
            ``ceil(message_length / 8)``
        _bch_ecc_bits : int
            BCH parity bit count (0 if ECC disabled).
        _crc_bits : int
            32 in combined mode, 0 otherwise.
        _total_ecc_bits : int
            ``_bch_ecc_bits + _crc_bits``
        _total_embedded : int
            Total bits written into the image:
            ``message_length + _total_ecc_bits × repeat`` (ECC on)
            or ``message_length × repeat`` (ECC off).
        """
        self._msg_byte_len = (self.message_length + 7) // 8

        use_ecc = self.use_error_correction and _BCH_AVAILABLE
        if use_ecc:
            # BCH(t=4, m=8): GF(2^8), corrects ≤4 bit errors per codeword.
            # Parity = m × t = 32 bits = 4 bytes, independent of data length.
            try:
                self._bch = bchlib.BCH(t=4, m=8)   # modern API
            except TypeError:
                self._bch = bchlib.BCH(8219, 4)    # legacy API
            self._bch_ecc_bits = self._bch.ecc_bytes * 8
        else:
            self._bch = None
            self._bch_ecc_bits = 0

        self._crc_bits       = 32 if (use_ecc and self.ecc_mode == "combined") else 0
        self._total_ecc_bits = self._bch_ecc_bits + self._crc_bits
        self._total_embedded = (
            self.message_length + self._total_ecc_bits * self.repeat
            if use_ecc else
            self.message_length * self.repeat
        )

    def _build_neighbor_offsets(self) -> None:
        """
        Precompute spatial offsets for the k×k neighbourhood ring.

        For a k×k window (k = ``neighborhood_size``) every cell except the
        center is a neighbour.  Offsets are stored as signed integer arrays
        so they can be broadcast directly onto index arrays in :meth:`decode`.

        Attributes set
        --------------
        _DR : np.ndarray[intp], shape (k²−1,)
            Row offsets of every neighbour relative to the center.
        _DC : np.ndarray[intp], shape (k²−1,)
            Column offsets of every neighbour relative to the center.
        _pad : int
            Padding width required for safe neighbourhood access
            (= ``neighborhood_size // 2``).

        Examples
        --------
        3×3  →  8 neighbours,  _pad = 1
        5×5  → 24 neighbours,  _pad = 2
        """
        half = self.neighborhood_size // 2
        # All (row, col) offsets in [-half, half]² — shape (k, k) each
        dr, dc = np.mgrid[-half : half + 1, -half : half + 1]
        dr, dc = dr.ravel(), dc.ravel()
        # Drop the center offset (0, 0)
        mask      = (dr != 0) | (dc != 0)
        self._DR  = dr[mask].astype(np.intp)
        self._DC  = dc[mask].astype(np.intp)
        self._pad = half

    # ------------------------------------------------------------------ #
    #  Index generation                                                     #
    # ------------------------------------------------------------------ #

    def _generate_indices(
        self, H: int, W: int, C: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample ``_total_embedded`` unique pixel positions from the full
        ``H × W × C`` space using a seeded Generator.

        ``np.unravel_index`` converts the flat sample to ``(rows, cols, chans)``
        in a single vectorized call — no arithmetic loops.

        Reflect-padding in :meth:`decode` handles any border pixels, so the
        full pixel space is available without exclusions.

        Parameters
        ----------
        H, W, C : int
            Image dimensions.

        Returns
        -------
        rows : np.ndarray[intp], shape (_total_embedded,)
        cols : np.ndarray[intp], shape (_total_embedded,)
        chans : np.ndarray[intp], shape (_total_embedded,)

        Raises
        ------
        ValueError
            If ``_total_embedded > H × W × C``.
        """
        n = self._total_embedded
        if n > H * W * C:
            raise ValueError(
                f"Image too small: need {n} positions, "
                f"capacity is {H * W * C} ({H}×{W}×{C})."
            )
        flat = np.random.default_rng(self.key).choice(
            H * W * C, size=n, replace=False
        )
        return np.unravel_index(flat, (H, W, C))

    # ------------------------------------------------------------------ #
    #  ECC encode / decode                                                  #
    # ------------------------------------------------------------------ #

    def _ecc_encode(self, message_bits: np.ndarray) -> np.ndarray:
        """
        Compute ECC bits for *message_bits*.

        ``'combined'`` mode: CRC-32 of the message is appended before BCH
        encoding; returned layout is ``[CRC-32 bits | BCH parity bits]``.

        Parameters
        ----------
        message_bits : np.ndarray, shape (message_length,), dtype uint8

        Returns
        -------
        np.ndarray, shape (_total_ecc_bits,), dtype uint8
        """
        msg_bytes = bytearray(
            np.packbits(message_bits).tobytes()[: self._msg_byte_len]
        )

        if self.ecc_mode == "combined":
            crc_val   = binascii.crc32(bytes(msg_bytes)) & 0xFFFFFFFF
            crc_bytes = bytearray(crc_val.to_bytes(4, "big"))
            bch_ecc   = bytearray(self._bch.encode(msg_bytes + crc_bytes))
            return np.concatenate([
                np.unpackbits(np.frombuffer(crc_bytes, dtype=np.uint8)),
                np.unpackbits(np.frombuffer(bch_ecc,   dtype=np.uint8)),
            ])

        return np.unpackbits(
            np.frombuffer(bytearray(self._bch.encode(msg_bytes)), dtype=np.uint8)
        )

    def _ecc_decode(
        self,
        noisy_msg_bits: np.ndarray,
        voted_ecc_bits: np.ndarray,
    ) -> np.ndarray:
        """
        Attempt BCH correction of *noisy_msg_bits* using *voted_ecc_bits*.

        Parameters
        ----------
        noisy_msg_bits : np.ndarray, shape (message_length,), dtype uint8
            Raw (possibly corrupted) message bits from the image.
        voted_ecc_bits : np.ndarray, shape (_total_ecc_bits,), dtype uint8
            Majority-voted ECC bits.

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
            Corrected bits, or *noisy_msg_bits* unchanged if BCH fails
            or CRC mismatches (combined mode).
        """
        msg_bytes = bytearray(
            np.packbits(noisy_msg_bits).tobytes()[: self._msg_byte_len]
        )

        if self.ecc_mode == "combined":
            crc_bytes = bytearray(
                np.packbits(voted_ecc_bits[: self._crc_bits]).tobytes()[:4]
            )
            bch_ecc = bytearray(
                np.packbits(voted_ecc_bits[self._crc_bits :]).tobytes()[: self._bch.ecc_bytes]
            )
            try:
                nerr, corrected, _ = self._bch.decode(msg_bytes + crc_bytes, bch_ecc)
            except Exception:
                return noisy_msg_bits
            if nerr < 0:
                return noisy_msg_bits
            corr_msg = bytes(corrected[: self._msg_byte_len])
            corr_crc = bytes(corrected[self._msg_byte_len : self._msg_byte_len + 4])
            if (binascii.crc32(corr_msg) & 0xFFFFFFFF) != int.from_bytes(corr_crc, "big"):
                return noisy_msg_bits   # integrity check failed
            return np.unpackbits(np.frombuffer(corr_msg, dtype=np.uint8))[: self.message_length]

        # single mode
        bch_ecc = bytearray(
            np.packbits(voted_ecc_bits).tobytes()[: self._bch.ecc_bytes]
        )
        try:
            nerr, corrected, _ = self._bch.decode(msg_bytes, bch_ecc)
        except Exception:
            return noisy_msg_bits
        if nerr < 0:
            return noisy_msg_bits
        return np.unpackbits(
            np.frombuffer(bytes(corrected[: self._msg_byte_len]), dtype=np.uint8)
        )[: self.message_length]

    # ------------------------------------------------------------------ #
    #  Public API                                                           #
    # ------------------------------------------------------------------ #

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """
        Embed *watermark* into *image* via additive pixel modulation.

        Payload layout::

            ECC disabled : tile(message, repeat)
            ECC enabled  : concat(message × 1,  tile(ecc_bits, repeat))

        Embedding rule (no Python loops)::

            img[row_i, col_i, chan_i] += alpha × (2 × bit_i − 1)

        Output is clipped to [0, 255].

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C), values in [0, 255]
            Source image (uint8 or float).
        watermark : np.ndarray, shape (message_length,), dtype uint8
            Binary watermark bits (0 or 1).

        Returns
        -------
        np.ndarray
            Watermarked image — same shape and dtype as *image*.
        """
        orig_dtype = image.dtype
        grayscale  = image.ndim == 2

        img = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        wm = np.array(watermark, dtype=np.uint8)
        payload = (
            np.concatenate([wm, np.tile(self._ecc_encode(wm), self.repeat)])
            if self._bch is not None else
            np.tile(wm, self.repeat)
        )

        rows, cols, chans = self._generate_indices(H, W, C)
        # ±alpha: maps bit ∈ {0,1} → delta ∈ {-alpha, +alpha}
        img[rows, cols, chans] += self.alpha * (2.0 * payload.astype(np.float32) - 1.0)
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, watermarked_or_attacked_image: np.ndarray) -> np.ndarray:
        """
        Extract the binary watermark from *watermarked_or_attacked_image*.

        Only the ``_total_embedded`` selected pixels and their immediate
        k×k neighbours are ever read — no full-image scan.

        Decision rule::

            center_i     = img[row_i, col_i, chan_i]
            neighbors_i  = k×k ring pixels at (row_i, col_i), same channel
            local_mean_i = mean(neighbors_i)          # center excluded
            raw_bit_i    = 1  if center_i ≥ local_mean_i  else 0

        Reflect-padding (width = ``neighborhood_size // 2``) makes border
        pixels safe without excluding any indices.

        Aggregation::

            ECC disabled : majority vote over (repeat, message_length) grid
            ECC enabled  : majority-vote ECC copies → BCH_correct(msg bits)

        Parameters
        ----------
        watermarked_or_attacked_image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
            Decoded binary watermark.
        """
        grayscale = watermarked_or_attacked_image.ndim == 2
        img = watermarked_or_attacked_image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        rows, cols, chans = self._generate_indices(H, W, C)

        # Reflect-pad by _pad pixels on each spatial side.
        # After padding: img[r, c, ch]  ≡  padded[r + _pad, c + _pad, ch]
        padded = np.pad(img, ((self._pad, self._pad), (self._pad, self._pad), (0, 0)),
                        mode="reflect")

        # ── Vectorized neighbourhood extraction ─────────────────────────
        # Only the _total_embedded selected pixels are processed.
        #
        # nr, nc : (n_bits, n_neighbors) — absolute indices into *padded*
        #   rows/cols are 0-based in img → add _pad to convert to padded coords.
        nr  = rows[:, np.newaxis] + self._pad + self._DR   # (n_bits, n_neighbors)
        nc  = cols[:, np.newaxis] + self._pad + self._DC   # (n_bits, n_neighbors)
        nch = np.broadcast_to(chans[:, np.newaxis], nr.shape).astype(np.intp)

        # Single fancy-index call: extracts all neighbourhoods at once.
        local_mean = padded[nr, nc, nch].mean(axis=1)           # (n_bits,)
        raw_bits   = (img[rows, cols, chans] >= local_mean).astype(np.uint8)

        # ── Aggregation ──────────────────────────────────────────────────
        if self._bch is not None:
            ecc_raw   = raw_bits[self.message_length :].reshape(
                self.repeat, self._total_ecc_bits
            )
            voted_ecc = (ecc_raw.sum(axis=0) * 2 >= self.repeat).astype(np.uint8)
            return self._ecc_decode(raw_bits[: self.message_length], voted_ecc)

        repeated = raw_bits.reshape(self.repeat, self.message_length)
        return (repeated.sum(axis=0) * 2 >= self.repeat).astype(np.uint8)
