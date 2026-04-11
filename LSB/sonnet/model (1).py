"""
model.py
========
Highly optimized, fully vectorized blind watermarking system.

Architecture
------------
Encoding
    Each bit is embedded by shifting a selected pixel by ±alpha::

        bit = 1  →  pixel += alpha
        bit = 0  →  pixel -= alpha

Decoding
    For every embedded position the center pixel is compared to the mean
    of its 8 spatial neighbours in the same channel (center excluded).
    The image is symmetrically padded before neighbourhood extraction so
    border positions are handled transparently — no positions need to be
    excluded from the index pool::

        center ≥ local_mean  →  1
        center <  local_mean  →  0

Bit layout
----------
ECC disabled::

    payload = tile(message, repeat)               # (message_length × repeat,)
    → majority vote per bit position at decode time

ECC enabled (BCH)::

    payload = concat( message×1 , tile(ecc, repeat) )
    #                  ^once          ^repeat times

    The message is embedded exactly once.
    BCH parity bits are tiled *repeat* times, majority-voted at decode time,
    then fed to the BCH decoder to correct errors in the single message copy.
    Combined mode additionally prepends a CRC-32 of the message to the data
    protected by BCH, providing a post-correction integrity check.

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

# 8-connected neighbour offsets for the 3×3 ring (center excluded), shape (8,).
_DR = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.intp)
_DC = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.intp)


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
        Seed for the deterministic PRNG that selects embedding positions.
        Must be identical at encode and decode time.
    alpha : float, default 4.0
        Embedding strength. Larger values improve robustness but reduce
        visual quality (lower PSNR).
    repeat : int, default 5
        ECC disabled : the full message is tiled *repeat* times;
        majority voting recovers each bit.
        ECC enabled  : only the BCH parity bits are tiled *repeat* times;
        the message itself is embedded exactly once.
    use_error_correction : bool, default True
        Enable BCH error correction (requires ``bchlib``).
        Falls back silently to pure repetition if the library is absent.
    ecc_mode : str, default ``'single'``
        ``'single'``   — BCH parity protection only.
        ``'combined'`` — CRC-32 of the message is appended to the data
                         that BCH protects, enabling post-correction
                         integrity verification.
    """

    def __init__(
        self,
        message_length: int    = 32,
        psnr_threshold: float  = 30.0,
        max_encode_time: float = 5.0,
        max_decode_time: float = 1.0,
        key: int               = 42,
        alpha: float           = 4.0,
        repeat: int            = 5,
        use_error_correction: bool = True,
        ecc_mode: str          = "single",
    ) -> None:
        self.message_length       = message_length
        self.psnr_threshold       = psnr_threshold
        self.max_encode_time      = max_encode_time
        self.max_decode_time      = max_decode_time
        self.key                  = key
        self.alpha                = float(alpha)
        self.repeat               = repeat
        self.use_error_correction = use_error_correction
        self.ecc_mode             = ecc_mode

        self._setup_ecc()

    # ------------------------------------------------------------------ #
    #  ECC initialisation                                                   #
    # ------------------------------------------------------------------ #

    def _setup_ecc(self) -> None:
        """
        Initialise the BCH codec and precompute bit-layout constants.

        Attributes set
        --------------
        _bch : bchlib.BCH or None
        _msg_byte_len : int
            ``ceil(message_length / 8)`` — byte length of the padded message.
        _bch_ecc_bits : int
            BCH parity bit count (0 when ECC is disabled).
        _crc_bits : int
            CRC-32 bit count: 32 in combined mode, 0 otherwise.
        _total_ecc_bits : int
            ``_bch_ecc_bits + _crc_bits``
        _total_embedded : int
            Total number of bits written into the image::

                ECC enabled  : message_length + _total_ecc_bits × repeat
                ECC disabled : message_length × repeat
        """
        self._msg_byte_len = (self.message_length + 7) // 8

        use_ecc = self.use_error_correction and _BCH_AVAILABLE
        if use_ecc:
            # BCH(t=4, m=8): GF(2^8), corrects up to 4 bit errors per codeword.
            # Parity = m × t = 32 bits = 4 bytes, regardless of data length.
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

    # ------------------------------------------------------------------ #
    #  Index generation                                                     #
    # ------------------------------------------------------------------ #

    def _generate_indices(
        self, H: int, W: int, C: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate ``_total_embedded`` unique pixel positions across the full
        ``H × W × C`` pixel space.

        All pixels are valid candidates. Border positions produce
        out-of-bounds neighbours, which are handled transparently by the
        symmetric reflect-padding applied in :meth:`decode`.

        The flat index space is laid out as::

            flat = chan + C × (col + W × row)

        Parameters
        ----------
        H, W, C : int
            Image height, width, number of channels.

        Returns
        -------
        rows : np.ndarray[intp], shape (_total_embedded,)
        cols : np.ndarray[intp], shape (_total_embedded,)
        chans : np.ndarray[intp], shape (_total_embedded,)

        Raises
        ------
        ValueError
            If the image is too small to hold ``_total_embedded`` bits.
        """
        n = self._total_embedded
        if n > H * W * C:
            raise ValueError(
                f"Image too small: need {n} positions, "
                f"image capacity is {H * W * C} pixels ({H}×{W}×{C})."
            )
        flat  = np.random.RandomState(self.key).choice(H * W * C, size=n, replace=False)
        chans = (flat % C).astype(np.intp)
        cols  = (flat // C % W).astype(np.intp)
        rows  = (flat // C // W).astype(np.intp)
        return rows, cols, chans

    # ------------------------------------------------------------------ #
    #  ECC encode / decode                                                  #
    # ------------------------------------------------------------------ #

    def _ecc_encode(self, message_bits: np.ndarray) -> np.ndarray:
        """
        Compute ECC bits for *message_bits*.

        In ``'combined'`` mode the CRC-32 of the message is appended to the
        message bytes before BCH encoding; the returned array is
        ``[CRC-32 bits | BCH parity bits]``.

        Parameters
        ----------
        message_bits : np.ndarray, shape (message_length,), dtype uint8

        Returns
        -------
        np.ndarray, shape (_total_ecc_bits,), dtype uint8
            Empty when ECC is disabled.
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
        Attempt BCH correction on *noisy_msg_bits* using *voted_ecc_bits*.

        Parameters
        ----------
        noisy_msg_bits : np.ndarray, shape (message_length,), dtype uint8
            Raw (possibly corrupted) message bits extracted from the image.
        voted_ecc_bits : np.ndarray, shape (_total_ecc_bits,), dtype uint8
            Majority-voted ECC bits.

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
            BCH-corrected bits, or *noisy_msg_bits* unchanged if correction
            fails (uncorrectable error pattern or CRC mismatch).
        """
        msg_bytes = bytearray(
            np.packbits(noisy_msg_bits).tobytes()[: self._msg_byte_len]
        )

        if self.ecc_mode == "combined":
            crc_bytes = bytearray(np.packbits(voted_ecc_bits[: self._crc_bits]).tobytes()[:4])
            bch_ecc   = bytearray(
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
            # Integrity gate: recompute CRC and compare
            if (binascii.crc32(corr_msg) & 0xFFFFFFFF) != int.from_bytes(corr_crc, "big"):
                return noisy_msg_bits
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

        Embedding rule (vectorized, no Python loops)::

            img[row_i, col_i, chan_i] += alpha × (2 × bit_i − 1)

        Values are clipped to [0, 255] after embedding.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C), dtype uint8 or float
            Source image with pixel values in [0, 255].
        watermark : np.ndarray, shape (message_length,), dtype uint8
            Binary watermark bits (values 0 or 1).

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

        wm      = watermark.astype(np.uint8)
        payload = (
            np.concatenate([wm, np.tile(self._ecc_encode(wm), self.repeat)])
            if self._bch is not None else
            np.tile(wm, self.repeat)
        )

        rows, cols, chans = self._generate_indices(H, W, C)
        img[rows, cols, chans] += self.alpha * (2.0 * payload.astype(np.float32) - 1.0)
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, watermarked_or_attacked_image: np.ndarray) -> np.ndarray:
        """
        Extract the binary watermark from *watermarked_or_attacked_image*.

        Per-bit decision (fully vectorized)::

            center_i     = img[row_i, col_i, chan_i]
            neighbors_i  = 3×3 ring around (row_i, col_i) in channel chan_i
            local_mean_i = mean(neighbors_i)   # center NOT included
            raw_bit_i    = 1  if center_i ≥ local_mean_i  else 0

        The image is symmetrically reflect-padded before extraction so that
        border pixels are handled without any special-casing.

        Aggregation::

            ECC disabled : majority_vote( raw_bits reshaped to (repeat, msg_len) )
            ECC enabled  : majority_vote ECC copies → BCH_correct(raw msg bits)

        Parameters
        ----------
        watermarked_or_attacked_image : np.ndarray, shape (H, W) or (H, W, C)
            Watermarked (possibly attacked) image.

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
            Decoded binary watermark bits.
        """
        grayscale = watermarked_or_attacked_image.ndim == 2
        img = watermarked_or_attacked_image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        rows, cols, chans = self._generate_indices(H, W, C)

        # Reflect-pad: border pixels get a valid 3×3 ring transparently.
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="reflect")

        # Vectorized neighbourhood extraction — single fancy-index call.
        # +1 offset in nr/nc accounts for the padding shift.
        nr  = rows[:, np.newaxis]  + 1 + _DR          # (n_bits, 8)
        nc  = cols[:, np.newaxis]  + 1 + _DC          # (n_bits, 8)
        nch = np.broadcast_to(chans[:, np.newaxis], nr.shape).astype(np.intp)

        local_mean = padded[nr, nc, nch].mean(axis=1)               # (n_bits,)
        raw_bits   = (img[rows, cols, chans] >= local_mean).astype(np.uint8)

        if self._bch is not None:
            ecc_raw   = raw_bits[self.message_length :].reshape(self.repeat, self._total_ecc_bits)
            voted_ecc = (ecc_raw.sum(axis=0) * 2 >= self.repeat).astype(np.uint8)
            return self._ecc_decode(raw_bits[: self.message_length], voted_ecc)

        repeated = raw_bits.reshape(self.repeat, self.message_length)
        return (repeated.sum(axis=0) * 2 >= self.repeat).astype(np.uint8)
