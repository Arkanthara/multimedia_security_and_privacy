"""
model.py
========
Highly optimized, fully vectorized blind watermarking system.

Architecture
------------
Encoding
    Each bit is embedded by shifting a selected pixel by ±alpha:
        bit = 1  →  pixel += alpha
        bit = 0  →  pixel -= alpha
    Pixel positions are drawn pseudo-randomly from the image interior
    (1-pixel border excluded) so that every selected pixel always has a
    valid 3×3 neighbourhood at decode time.

Decoding
    For every embedded position the center pixel is compared to the mean
    of its 8 spatial neighbours **in the same channel** (center excluded).
        center ≥ local_mean  →  1
        center <  local_mean  →  0
    All neighbourhood extractions are fully vectorized via fancy indexing
    on a symmetrically-padded copy of the image.

Bit layout (ECC disabled)
    [ message × repeat ]                  → majority vote per bit position

Bit layout (ECC enabled – BCH)
    [ message (×1) | BCH-parity × repeat ]
    The message is embedded exactly once.  The BCH parity bits are embedded
    *repeat* times and majority-voted before being fed to the BCH decoder,
    which corrects residual bit errors in the single message copy.

    Combined mode prepends a CRC-32 of the message to the data that BCH
    protects, providing an extra post-correction integrity check.

Dependencies
------------
Required  : numpy, opencv-python
Optional  : bchlib  (ECC; graceful fallback to pure repetition if absent)
"""

from __future__ import annotations

import binascii
from typing import Tuple

import cv2          # noqa: F401 – available for caller convenience / PSNR
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Optional BCH dependency
# ──────────────────────────────────────────────────────────────────────────────
try:
    import bchlib
    _BCH_AVAILABLE = True
except ImportError:
    _BCH_AVAILABLE = False

# Neighbour offsets for the 8-connected 3×3 ring (center excluded), row-major.
# Shape (2, 8) — first row = Δrow, second row = Δcol.
_NEIGH_DR = np.array([-1, -1, -1,  0,  0,  1,  1,  1], dtype=np.intp)
_NEIGH_DC = np.array([-1,  0,  1, -1,  1, -1,  0,  1], dtype=np.intp)


# ──────────────────────────────────────────────────────────────────────────────
#  WatermarkModel
# ──────────────────────────────────────────────────────────────────────────────

class WatermarkModel:
    """
    Vectorized blind watermarking model using local pixel modulation.

    Parameters
    ----------
    message_length : int, default 32
        Number of bits in the watermark message.
    psnr_threshold : float, default 30.0
        Minimum acceptable PSNR (dB).  Informational only.
    max_encode_time : float, default 5.0
        Encoding time budget in seconds.  Informational only.
    max_decode_time : float, default 1.0
        Decoding time budget in seconds.  Informational only.
    key : int, default 42
        Seed for the deterministic PRNG that selects embedding positions.
        Must be identical at encode and decode time.
    alpha : float, default 4.0
        Embedding strength.  Larger values improve robustness but reduce
        visual quality (lower PSNR).
    repeat : int, default 5
        - ECC disabled : the full message is tiled *repeat* times;
          majority voting recovers each bit.
        - ECC enabled  : only the BCH parity bits are tiled *repeat* times;
          the message itself is embedded exactly once.
    use_error_correction : bool, default True
        Enable BCH error correction (requires ``bchlib``).
        Falls back silently to pure repetition if the library is absent.
    ecc_mode : str, default ``'single'``
        ``'single'``   — BCH parity only.
        ``'combined'`` — CRC-32 of the message is appended to the data
                         protected by BCH, enabling post-correction
                         integrity verification.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                         #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        message_length: int   = 32,
        psnr_threshold: float = 30.0,
        max_encode_time: float = 5.0,
        max_decode_time: float = 1.0,
        key: int              = 42,
        alpha: float          = 4.0,
        repeat: int           = 5,
        use_error_correction: bool = True,
        ecc_mode: str         = "single",
    ) -> None:
        self.message_length      = message_length
        self.psnr_threshold      = psnr_threshold
        self.max_encode_time     = max_encode_time
        self.max_decode_time     = max_decode_time
        self.key                 = key
        self.alpha               = float(alpha)
        self.repeat              = repeat
        self.use_error_correction = use_error_correction
        self.ecc_mode            = ecc_mode

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
            ``ceil(message_length / 8)`` — byte length of the message.
        _bch_ecc_bits : int
            BCH parity bit count (0 when ECC is disabled).
        _crc_bits : int
            CRC-32 bit count (32 in combined mode, else 0).
        _total_ecc_bits : int
            ``_bch_ecc_bits + _crc_bits``
        _total_embedded : int
            Total number of bits written into the image.
        """
        self._msg_byte_len = (self.message_length + 7) // 8

        use_ecc = self.use_error_correction and _BCH_AVAILABLE
        if use_ecc:
            # BCH(t=4, m=8): GF(2^8), corrects up to 4 bit errors per
            # codeword.  Parity = m*t = 32 bits = 4 bytes, regardless of
            # data length (up to 223 bits for GF(2^8), t=4).
            try:
                self._bch = bchlib.BCH(t=4, m=8)           # modern API
            except TypeError:
                self._bch = bchlib.BCH(8219, 4)            # legacy API
            self._bch_ecc_bits = self._bch.ecc_bytes * 8
        else:
            self._bch = None
            self._bch_ecc_bits = 0

        self._crc_bits      = 32 if (use_ecc and self.ecc_mode == "combined") else 0
        self._total_ecc_bits = self._bch_ecc_bits + self._crc_bits

        if use_ecc:
            # Layout: [ message (×1) | ecc_bits × repeat ]
            self._total_embedded = self.message_length + self._total_ecc_bits * self.repeat
        else:
            # Layout: [ message × repeat ]
            self._total_embedded = self.message_length * self.repeat

    # ------------------------------------------------------------------ #
    #  Index generation                                                     #
    # ------------------------------------------------------------------ #

    def _generate_indices(
        self, H: int, W: int, C: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate ``_total_embedded`` unique embedding positions inside the
        image interior (1-pixel border excluded on all sides).

        The interior is treated as a flat array of size
        ``(H-2) × (W-2) × C``, indexed as::

            flat = chan + C * (col_interior + (W-2) * row_interior)

        where ``row_interior ∈ [0, H-3]``, ``col_interior ∈ [0, W-3]``.
        The border exclusion guarantees that every selected pixel has a
        full 3×3 neighbourhood without any out-of-bounds access.

        Parameters
        ----------
        H, W, C : int
            Image height, width, number of channels.

        Returns
        -------
        rows : np.ndarray[intp], shape (_total_embedded,)
            Row indices in [1, H-2].
        cols : np.ndarray[intp], shape (_total_embedded,)
            Column indices in [1, W-2].
        chans : np.ndarray[intp], shape (_total_embedded,)
            Channel indices in [0, C-1].

        Raises
        ------
        ValueError
            If the image interior is too small to hold all required bits.
        """
        n = self._total_embedded
        interior_capacity = (H - 2) * (W - 2) * C
        if n > interior_capacity:
            raise ValueError(
                f"Image interior too small: need {n} positions, "
                f"capacity is {interior_capacity} pixels "
                f"(image {H}×{W}×{C} with 1-pixel border excluded)."
            )

        rng  = np.random.RandomState(self.key)
        flat = rng.choice(interior_capacity, size=n, replace=False)

        # Vectorized decomposition — zero Python loops
        chans = (flat % C).astype(np.intp)
        rc    = flat // C                            # index in (H-2)×(W-2) grid
        rows  = (rc // (W - 2) + 1).astype(np.intp) # shift interior→image coords
        cols  = (rc %  (W - 2) + 1).astype(np.intp)
        return rows, cols, chans

    # ------------------------------------------------------------------ #
    #  ECC helpers                                                          #
    # ------------------------------------------------------------------ #

    def _ecc_encode(self, message_bits: np.ndarray) -> np.ndarray:
        """
        Compute ECC bits for *message_bits*.

        In ``'combined'`` mode the CRC-32 of the message is appended to the
        message bytes before BCH encoding; both CRC bits and BCH parity bits
        are returned concatenated (CRC first).

        Parameters
        ----------
        message_bits : np.ndarray, shape (message_length,), dtype uint8
            Binary message (values 0 or 1).

        Returns
        -------
        ecc_bits : np.ndarray, shape (_total_ecc_bits,), dtype uint8
            Empty array when ECC is disabled.
        """
        if self._bch is None:
            return np.empty(0, dtype=np.uint8)

        msg_bytes = bytearray(
            np.packbits(message_bits).tobytes()[: self._msg_byte_len]
        )

        if self.ecc_mode == "combined":
            crc_val   = binascii.crc32(bytes(msg_bytes)) & 0xFFFFFFFF
            crc_bytes = bytearray(crc_val.to_bytes(4, "big"))
            bch_ecc   = bytearray(self._bch.encode(msg_bytes + crc_bytes))
            crc_bits  = np.unpackbits(np.frombuffer(crc_bytes, dtype=np.uint8))
            bch_bits  = np.unpackbits(np.frombuffer(bch_ecc,   dtype=np.uint8))
            # Layout: [ CRC-32 bits | BCH parity bits ]
            return np.concatenate([crc_bits, bch_bits])

        # single mode
        bch_ecc = bytearray(self._bch.encode(msg_bytes))
        return np.unpackbits(np.frombuffer(bch_ecc, dtype=np.uint8))

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
            BCH-corrected bits, or the original *noisy_msg_bits* if BCH
            correction fails (uncorrectable error pattern).
        """
        if self._bch is None:
            return noisy_msg_bits

        msg_bytes = bytearray(
            np.packbits(noisy_msg_bits).tobytes()[: self._msg_byte_len]
        )

        if self.ecc_mode == "combined":
            crc_bits  = voted_ecc_bits[: self._crc_bits]
            bch_bits  = voted_ecc_bits[self._crc_bits :]
            crc_bytes = bytearray(np.packbits(crc_bits).tobytes()[:4])
            bch_ecc   = bytearray(
                np.packbits(bch_bits).tobytes()[: self._bch.ecc_bytes]
            )
            data = msg_bytes + crc_bytes
            try:
                nerr, corrected, _ = self._bch.decode(data, bch_ecc)
            except Exception:
                return noisy_msg_bits
            if nerr < 0:
                return noisy_msg_bits
            corr_msg = bytes(corrected[: self._msg_byte_len])
            corr_crc = bytes(corrected[self._msg_byte_len : self._msg_byte_len + 4])
            # Post-correction CRC integrity check
            if (binascii.crc32(corr_msg) & 0xFFFFFFFF) == int.from_bytes(corr_crc, "big"):
                return np.unpackbits(
                    np.frombuffer(corr_msg, dtype=np.uint8)
                )[: self.message_length]
            return noisy_msg_bits  # CRC mismatch → correction unreliable

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

        Full bit payload
        ----------------
        ECC disabled::

            payload = tile(message, repeat)            # shape: (msg_len × repeat,)

        ECC enabled::

            payload = concat(message, tile(ecc, repeat))
            #                ^once        ^repeat times

        Embedding rule (vectorized, no Python loops)::

            img[row_i, col_i, chan_i] += alpha * (2 * bit_i - 1)

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C), dtype uint8 or float
            Source image.  Pixel values must lie in [0, 255].
        watermark : np.ndarray, shape (message_length,), dtype uint8
            Binary watermark bits (values 0 or 1).

        Returns
        -------
        np.ndarray
            Watermarked image — same shape and dtype as *image*, clipped to
            [0, 255].
        """
        orig_dtype = image.dtype
        grayscale  = image.ndim == 2

        # Work internally as float32, shape (H, W, C)
        img = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        wm = watermark.astype(np.uint8)

        # ── Build payload ────────────────────────────────────────────────
        if self._bch is not None:
            ecc_bits = self._ecc_encode(wm)
            payload  = np.concatenate([wm, np.tile(ecc_bits, self.repeat)])
        else:
            payload  = np.tile(wm, self.repeat)

        # ── Embed (fully vectorized) ─────────────────────────────────────
        rows, cols, chans = self._generate_indices(H, W, C)

        # ±alpha via the identity: 2*bit - 1 ∈ {-1, +1}
        delta = self.alpha * (2.0 * payload.astype(np.float32) - 1.0)
        img[rows, cols, chans] += delta
        np.clip(img, 0.0, 255.0, out=img)      # prevent overflow/underflow

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, watermarked_or_attacked_image: np.ndarray) -> np.ndarray:
        """
        Extract the binary watermark from *watermarked_or_attacked_image*.

        Per-bit decision (fully vectorized)
        ------------------------------------
        For every embedded position *i*::

            center_i    = img[row_i, col_i, chan_i]
            neighbors_i = 3×3 ring around (row_i, col_i) in channel chan_i
            local_mean_i = mean(neighbors_i)           # center NOT included
            raw_bit_i    = 1  if center_i >= local_mean_i  else 0

        Aggregation
        -----------
        ECC disabled — majority vote::

            bit_j = (Σ_k raw_bit_{j + k*msg_len} ≥ repeat/2)

        ECC enabled — vote ECC, then correct message::

            voted_ecc = majority_vote over *repeat* ECC copies
            message   = BCH_correct(raw message bits, voted_ecc)

        Parameters
        ----------
        watermarked_or_attacked_image : np.ndarray, shape (H, W) or (H, W, C)
            Watermarked (possibly attacked) image.

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
            Decoded binary watermark bits.

        Notes
        -----
        * Symmetric reflect-padding prevents border artefacts.
        * Neighbourhood extraction uses a single fancy-index call — no loops.
        """
        grayscale = watermarked_or_attacked_image.ndim == 2
        img = watermarked_or_attacked_image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        rows, cols, chans = self._generate_indices(H, W, C)

        # ── Symmetric padding (+1 on each spatial side) ──────────────────
        # Allows safe 3×3 access even for positions on the image edge.
        # (Interior-only index generation makes this redundant in practice,
        #  but it is kept as a safety net against future parameter changes.)
        padded = np.pad(img, ((1, 1), (1, 1), (0, 0)), mode="reflect")
        #               ↑ padded[r+1, c+1, ch]  corresponds to img[r, c, ch]

        # ── Center pixel values ──────────────────────────────────────────
        center = img[rows, cols, chans]      # (n_bits,)

        # ── Vectorized 3×3 neighbourhood extraction ──────────────────────
        # nr, nc : (n_bits, 8) — absolute indices into *padded*
        # +1 because padded has a 1-pixel shift relative to img
        nr  = rows[:, np.newaxis]  + 1 + _NEIGH_DR   # (n_bits, 8)
        nc  = cols[:, np.newaxis]  + 1 + _NEIGH_DC   # (n_bits, 8)
        nch = np.broadcast_to(
            chans[:, np.newaxis], (len(chans), 8)
        ).astype(np.intp)

        neighbors  = padded[nr, nc, nch]              # (n_bits, 8)
        local_mean = neighbors.mean(axis=1)           # (n_bits,)  — center excluded

        # ── Bit decision ─────────────────────────────────────────────────
        raw_bits = (center >= local_mean).astype(np.uint8)   # (n_bits,)

        # ── Aggregation ──────────────────────────────────────────────────
        if self._bch is not None:
            msg_bits  = raw_bits[: self.message_length]
            ecc_raw   = raw_bits[self.message_length :].reshape(
                self.repeat, self._total_ecc_bits
            )
            # Majority vote: count 1s across *repeat* copies, threshold at 50 %
            voted_ecc = (ecc_raw.sum(axis=0) * 2 >= self.repeat).astype(np.uint8)
            return self._ecc_decode(msg_bits, voted_ecc)

        # Majority vote over repeated message
        repeated = raw_bits.reshape(self.repeat, self.message_length)
        return (repeated.sum(axis=0) * 2 >= self.repeat).astype(np.uint8)
