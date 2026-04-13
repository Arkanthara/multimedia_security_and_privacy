"""
model.py
========
Fully vectorized blind watermarking — confidence-weighted decoding,
optional CRC-32 error correction.

Embedding rule
--------------
Each watermark bit shifts one randomly-selected pixel by ±alpha::

    bit = 1  →  pixel += alpha
    bit = 0  →  pixel -= alpha

Confidence score (core decoding primitive)
------------------------------------------
For every embedded pixel position::

    score = center_pixel − mean(k×k neighbourhood, center excluded)

Positive → likely 1; negative → likely 0; |score| = confidence.
Averaging scores across repeated copies of the same bit improves SNR
proportionally to √(copies), strictly better than binary majority voting
because the full signal magnitude is preserved before the hard decision.

Payload layouts
---------------
CRC disabled  (``use_crc=False``)::

    payload = tile(message, msg_repeat)

    Decode: reshape scores to (msg_repeat, message_length), average, hard-decide.
    Simple, fast, no overhead.  Raise *msg_repeat* for more robustness.

CRC enabled   (``use_crc=True``)::

    payload = [ tile(message, msg_repeat)  |  tile(CRC-32, crc_repeat) ]
                    m copies                       n copies,  n > m

    The CRC field gets more copies than the message so it is recovered
    more reliably, giving the repair stage a trusted oracle.

    Decode pipeline:
      1. Average message scores over m copies → mean score per bit.
      2. Average CRC scores over n copies → hard-decode reference CRC.
      3. Hard-decide message from sign of mean scores.
      4. Verify CRC-32(message) == reference CRC → return on success.
      5. Brute-force repair: flip the k = 1 … *max_flip_bits* least-confident
         message bits (sorted by |mean score| ascending) until CRC passes.

Why CRC-32 even for a 32-bit message?
--------------------------------------
CRC-32 doubles the bit overhead for a 32-bit message, but remains the right
choice for two reasons:

* **Silent errors.** Without a verifiable reference you cannot know whether
  the decoded output is correct.  CRC makes errors detectable.

* **Brute-force fidelity.** The repair stage needs an oracle whose false-
  positive rate is negligible across all candidates tried.  With k=3 and a
  pool of 32 bits the search has ≤ 4 960 candidates; CRC-32 gives a false-
  acceptance probability of 4 960 × 2⁻³² ≈ 10⁻⁶.  CRC-16 (half the
  overhead) would push that to ~5 % — unacceptably high.

Dependencies
------------
numpy, opencv-python (cv2 available for callers; not used internally).
"""

from __future__ import annotations

import binascii
import itertools
from typing import Tuple

import cv2          # noqa: F401 — available for callers (e.g. PSNR checks)
import numpy as np


# ── Module-level CRC helper ───────────────────────────────────────────────────

_CRC_BITS: int = 32   # width of the CRC field (fixed)


def _crc32_bits(bits: np.ndarray) -> np.ndarray:
    """
    Return the CRC-32 of *bits* as a (32,) uint8 bit array (big-endian).

    Parameters
    ----------
    bits : np.ndarray, shape (n,), dtype uint8

    Returns
    -------
    np.ndarray, shape (32,), dtype uint8
    """
    crc = binascii.crc32(np.packbits(bits).tobytes()) & 0xFFFF_FFFF
    return np.unpackbits(np.frombuffer(crc.to_bytes(4, "big"), dtype=np.uint8))


# ── Model ─────────────────────────────────────────────────────────────────────

class WatermarkModel:
    """
    Vectorized blind watermarking — confidence-weighted, optionally CRC-protected.

    Parameters
    ----------
    message_length : int, default 32
        Number of bits in the watermark message.
    key : int, default 42
        PRNG seed for deterministic pixel selection.
        **Must be identical at encode and decode time.**
    alpha : float, default 75.0
        Embedding strength. Larger → more robust, lower PSNR.
    use_crc : bool, default False
        Enable CRC-32 error detection and brute-force correction.
        When *False*, the entire embedding budget is spent on message copies,
        yielding better raw SNR at the cost of no integrity guarantee.
        Compensate by raising *msg_repeat* (e.g. 8–10).
    msg_repeat : int, default 35
        Number of message copies embedded (m).
        Also the sole redundancy knob when ``use_crc=False``.
        When ``use_crc=True``, must satisfy ``msg_repeat < crc_repeat``.
    crc_repeat : int, default 5
        Number of CRC-32 copies embedded (n). Ignored when ``use_crc=False``.
        Must satisfy ``crc_repeat > msg_repeat``.
    neighborhood_size : int, default 3
        Side length of the square neighbourhood for the confidence score.
        Must be an odd integer ≥ 3 (e.g. 3 for 3×3, 5 for 5×5).
    max_flip_bits : int, default 3
        Maximum bits flipped per brute-force repair level (CRC mode only).
        Cost at level k: C(min(message_length, 32), k).
        k=3 → ≤ 4 960 CRC checks (negligible runtime).

    Total embedded bits
    -------------------
    ``use_crc=False`` :  ``message_length × msg_repeat``
    ``use_crc=True``  :  ``message_length × msg_repeat + 32 × crc_repeat``
    """

    def __init__(
        self,
        message_length: int    = 32,
        key: int               = 42,
        alpha: float           = 75.0,
        use_crc: bool          = False,
        msg_repeat: int        = 35,
        crc_repeat: int        = 5,
        neighborhood_size: int = 3,
        max_flip_bits: int     = 3,
    ) -> None:
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")
        if use_crc and crc_repeat <= msg_repeat:
            raise ValueError(
                f"crc_repeat ({crc_repeat}) must exceed msg_repeat ({msg_repeat}) "
                "so the CRC field is recovered more reliably than the message."
            )

        self.message_length    = message_length
        self.key               = key
        self.alpha             = float(alpha)
        self.use_crc           = use_crc
        self.msg_repeat        = msg_repeat
        self.crc_repeat        = crc_repeat
        self.neighborhood_size = neighborhood_size
        self.max_flip_bits     = max_flip_bits

        self._total_embedded: int = (
            message_length * msg_repeat + _CRC_BITS * crc_repeat
            if use_crc else
            message_length * msg_repeat
        )

        # ── Neighbourhood ring offsets — computed once at construction ────
        # All (row, col) offsets in the k×k window except the center (0, 0).
        half   = neighborhood_size // 2
        dr, dc = np.mgrid[-half : half + 1, -half : half + 1]
        dr, dc = dr.ravel(), dc.ravel()
        ring   = (dr != 0) | (dc != 0)
        self._DR:  np.ndarray = dr[ring].astype(np.intp)   # (k²−1,)
        self._DC:  np.ndarray = dc[ring].astype(np.intp)   # (k²−1,)
        self._pad: int        = half

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _pixel_indices(
        self, H: int, W: int, C: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample ``_total_embedded`` unique pixel positions from H×W×C.

        Uses a seeded ``numpy.random.default_rng`` — identical seed →
        identical positions at encode and decode time.

        Returns
        -------
        rows, cols, chans : np.ndarray[intp], each shape (_total_embedded,)

        Raises
        ------
        ValueError if the image is too small to hold the payload.
        """
        n   = self._total_embedded
        cap = H * W * C
        if n > cap:
            raise ValueError(
                f"Image too small: need {n} pixel positions, "
                f"capacity is {cap} ({H}×{W}×{C})."
            )
        flat = np.random.default_rng(self.key).choice(cap, size=n, replace=False)
        return np.unravel_index(flat, (H, W, C))  # type: ignore[return-value]

    def _confidence_scores(
        self,
        img: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        chans: np.ndarray,
    ) -> np.ndarray:
        """
        Compute real-valued confidence scores at all embedded positions.

        For each position i::

            score_i = img[row_i, col_i, chan_i]
                    − mean( k×k neighbourhood, center excluded )

        Reflect-padding (width = _pad) handles border pixels transparently.

        Returns
        -------
        np.ndarray, shape (_total_embedded,), dtype float32
            Positive → bit likely 1; negative → likely 0;
            larger |score| → higher confidence.
        """
        padded = np.pad(
            img,
            ((self._pad, self._pad), (self._pad, self._pad), (0, 0)),
            mode="reflect",
        )
        # Neighbour absolute indices into padded array: (n_bits, n_neighbors)
        nr  = rows[:, np.newaxis]  + self._pad + self._DR
        nc  = cols[:, np.newaxis]  + self._pad + self._DC
        nch = np.broadcast_to(chans[:, np.newaxis], nr.shape).astype(np.intp)

        return img[rows, cols, chans] - padded[nr, nc, nch].mean(axis=1)

    def _crc_repair(
        self,
        msg_bits: np.ndarray,
        confidence: np.ndarray,
        ref_crc: np.ndarray,
    ) -> np.ndarray:
        """
        Flip the fewest, least-confident bits until CRC-32 passes.

        Levels k = 1, 2, …, *max_flip_bits* are tried in order.
        At level k all C(pool, k) combinations of flipping k bits drawn
        from the ``pool`` least-confident positions are tested.
        The pool is capped at 32 so the worst-case cost (k=3) is
        C(32, 3) = 4 960 CRC checks regardless of message length.

        Returns
        -------
        np.ndarray, shape (message_length,), uint8
            Corrected bits, or *msg_bits* unchanged if repair fails.
        """
        pool = np.argsort(confidence)[:min(self.message_length, 32)].tolist()

        for k in range(1, self.max_flip_bits + 1):
            for idx in itertools.combinations(pool, k):
                trial        = msg_bits.copy()
                trial[list(idx)] ^= np.uint8(1)
                if np.array_equal(_crc32_bits(trial), ref_crc):
                    return trial

        return msg_bits  # repair exhausted — return best-effort hard decision

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """
        Embed *watermark* into *image* via additive pixel modulation.

        Payload (fully vectorized)::

            use_crc=False : tile(message, msg_repeat)
            use_crc=True  : [tile(message, msg_repeat) | tile(CRC-32, crc_repeat)]

        Each selected pixel is modified by::

            Δ = alpha × (2 × bit − 1)      # +alpha for 1, −alpha for 0

        Output is clipped to [0, 255].

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C), uint8 or float
        watermark : np.ndarray, shape (message_length,), dtype uint8

        Returns
        -------
        np.ndarray — watermarked image, **same shape and dtype** as *image*.
        """
        orig_dtype = image.dtype
        grayscale  = image.ndim == 2
        img        = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        wm = np.asarray(watermark, dtype=np.uint8)
        payload = (
            np.concatenate([np.tile(wm, self.msg_repeat),
                            np.tile(_crc32_bits(wm), self.crc_repeat)])
            if self.use_crc else
            np.tile(wm, self.msg_repeat)
        )  # shape: (_total_embedded,)

        rows, cols, chans = self._pixel_indices(H, W, C)
        img[rows, cols, chans] += self.alpha * (2.0 * payload.astype(np.float32) - 1.0)
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the binary watermark from *image*.

        Only the ``_total_embedded`` selected pixels and their k×k neighbours
        are ever read — no full-image scan.

        CRC disabled
            Average confidence scores over ``msg_repeat`` copies of each bit;
            hard-decide from the sign of the mean.

        CRC enabled
            Same averaging for message bits, then for CRC bits.
            Verify CRC-32; on failure run brute-force repair on the
            least-confident message bits.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
        """
        grayscale = image.ndim == 2
        img       = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        rows, cols, chans = self._pixel_indices(H, W, C)
        scores = self._confidence_scores(img, rows, cols, chans)  # (_total_embedded,)

        # ── Average message scores over msg_repeat copies ────────────────
        msg_mean = (
            scores[: self.message_length * self.msg_repeat]
            .reshape(self.msg_repeat, self.message_length)
            .mean(axis=0)                                   # (message_length,)
        )
        msg_bits = (msg_mean >= 0.0).astype(np.uint8)       # hard decision
        msg_conf = np.abs(msg_mean)                          # per-bit confidence

        if not self.use_crc:
            return msg_bits

        # ── Average CRC scores over crc_repeat copies ────────────────────
        ref_crc = (
            scores[self.message_length * self.msg_repeat :]
            .reshape(self.crc_repeat, _CRC_BITS)
            .mean(axis=0)
            >= 0.0
        ).astype(np.uint8)                                   # (32,)

        # ── Verify → repair → return ─────────────────────────────────────
        if np.array_equal(_crc32_bits(msg_bits), ref_crc):
            return msg_bits

        return self._crc_repair(msg_bits, msg_conf, ref_crc)
