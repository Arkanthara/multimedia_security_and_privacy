"""
model.py
========
Vectorized blind watermarking — confidence-weighted decoding,
optional robustness to lossless geometric transforms (flips, 90° rotations).

Embedding rule
--------------
Each watermark bit shifts one randomly-selected pixel by ±alpha::

    bit = 1  →  pixel += alpha
    bit = 0  →  pixel -= alpha

Decoding — two complementary quality metrics
--------------------------------------------

**Metric 1 — Embedding strength** (``embedding_strength``)::

    strength = mean_i( |pixel_i − mean(k×k ring neighbourhood_i)| )

Measures how much the sampled pixels deviate from their local surroundings.
If those pixels were shifted by ±alpha at encode time, strength ≈ alpha,
regardless of the bit values (0 or 1 both give |Δ| = alpha).
Near zero → wrong pixel positions sampled (wrong orientation) or nothing
embedded.

**Metric 2 — Overall confidence** (``overall_confidence``)::

    For each bit position i ∈ [0, message_length):
        frac_i  = fraction of msg_repeat copies that decode bit i as 1
        conf_i  = max(frac_i, 1 − frac_i)       ∈ [0.5, 1.0]

    overall_confidence = mean_i( conf_i )        ∈ [0.5, 1.0]

Measures copy agreement at each bit position, then averages across bits.
  1.0 → every copy agrees on every bit — no copy was altered (perfect).
  0.5 → completely random (50/50 coin toss per bit across copies).

Key property: confidence is computed from copy *agreement*, not from bit
*values*.  It therefore works correctly for any watermark message, including
uniformly random ones (≈ 50 % ones) whose mean signed score is ≈ 0 — a case
that would fool a plain "mean-score magnitude" coherence test.

Transform-search strategy
--------------------------
When ``robust_to_transforms=True``, the decoder tries six lossless candidate
inverse transforms ordered cheapest-first::

    identity → hflip → vflip → rot270 → rot90 → rot180

All transforms are lossless (numpy views or ``numpy.rot90``): no interpolation,
no resolution change, and — for ±90° rotations — H and W simply swap so
``_pixel_indices`` adapts automatically.

For each candidate:
  1. Apply the inverse-transform candidate to the received image.
  2. Recompute pixel indices from the (possibly swapped) image shape.
  3. Rank by ``overall_confidence`` (primary criterion).
  4. **Early exit** if ``overall_confidence ≥ confidence_threshold``.
  5. After all candidates, return the highest-confidence result.

Dependencies
------------
numpy, opencv-python (cv2 re-exported for callers; not used internally).
"""

from __future__ import annotations

from typing import List, NamedTuple, Optional, Tuple

import cv2          # noqa: F401 — re-exported for callers (e.g. PSNR checks)
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Lossless transform catalogue
# ─────────────────────────────────────────────────────────────────────────────
# Each entry is the *inverse* of one possible attack transform.
# Ordered cheapest-first so the early-exit fires quickly for common cases.
#
#   Attack applied to image    Inverse tried here
#   ──────────────────────     ──────────────────
#   (none)                     identity
#   horizontal flip            hflip     (self-inverse)
#   vertical flip              vflip     (self-inverse)
#   90° CW rotation            rot270    (= 90° CCW)
#   90° CCW rotation           rot90     (= 90° CW)
#   180° rotation              rot180    (self-inverse)
# ─────────────────────────────────────────────────────────────────────────────

_CANDIDATE_TRANSFORMS: List[str] = [
    "identity",
    "hflip",
    "vflip",
    "rot270",
    "rot90",
    "rot180",
]


def _apply_transform(img: np.ndarray, name: str) -> np.ndarray:
    """
    Apply a named lossless geometric transform to a (H, W, C) float32 array.

    All operations are zero-copy views or single ``numpy.rot90`` calls —
    no interpolation, no quality loss.  For rot90 / rot270, H and W swap;
    for all others the shape is unchanged.

    Parameters
    ----------
    img  : np.ndarray, shape (H, W, C), float32
    name : str — one of the entries in ``_CANDIDATE_TRANSFORMS``

    Returns
    -------
    np.ndarray, shape (H', W', C), float32
    """
    if name == "identity":
        return img
    if name == "hflip":
        return img[:, ::-1, :]
    if name == "vflip":
        return img[::-1, :, :]
    if name == "rot90":                      # 90° CCW
        return np.rot90(img, k=1, axes=(0, 1))
    if name == "rot180":
        return np.rot90(img, k=2, axes=(0, 1))
    if name == "rot270":                     # 90° CW
        return np.rot90(img, k=3, axes=(0, 1))
    raise ValueError(f"Unknown transform: {name!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Public result type
# ─────────────────────────────────────────────────────────────────────────────

class DecodeResult(NamedTuple):
    """
    Full watermark decode result, returned by :meth:`WatermarkModel.decode_verbose`.

    Attributes
    ----------
    bits : np.ndarray, shape (message_length,), uint8
        Hard-decoded watermark bits (majority vote across ``msg_repeat`` copies).
    embedding_strength : float
        Mean |pixel − ring_mean| at selected positions (Metric 1).
        Scales with alpha; near zero → wrong orientation or nothing embedded.
    overall_confidence : float
        Mean per-bit copy-agreement fraction (Metric 2), in [0.5, 1.0].
        1.0 → every copy agrees on every bit (perfect decode).
        0.5 → fully random (wrong orientation or no watermark).
    transform : str
        Name of the candidate inverse transform that produced this result.
    """
    bits:               np.ndarray
    embedding_strength: float
    overall_confidence: float
    transform:          str


# ─────────────────────────────────────────────────────────────────────────────
# WatermarkModel
# ─────────────────────────────────────────────────────────────────────────────

class WatermarkModel:
    """
    Vectorized blind watermarking — confidence-weighted decoding,
    optionally robust to lossless geometric transforms.

    Parameters
    ----------
    message_length : int, default 32
        Number of bits in the watermark message.
    key : int, default 42
        PRNG seed for deterministic pixel selection.
        **Must be identical at encode and decode time.**
    alpha : float, default 75.0
        Embedding strength.  Larger → more robust, lower PSNR.
    msg_repeat : int, default 35
        Number of message copies embedded.  SNR ∝ √msg_repeat;
        raise (e.g. 50–100) for more robustness against post-processing.
    neighborhood_size : int, default 5
        Side length of the square neighbourhood used for confidence scores.
        Must be an odd integer ≥ 3 (e.g. 3 for 3×3, 5 for 5×5).
    robust_to_transforms : bool, default True
        When True, :meth:`decode` searches over all lossless candidate
        inverse transforms and returns the highest-confidence result.
    confidence_threshold : float, default 0.90
        Early-exit threshold for the transform search.
        A result with ``overall_confidence ≥ confidence_threshold`` is
        accepted immediately (range (0.5, 1.0]; 0.90 → 90 % copy agreement).

    Total embedded bits
    -------------------
    ``message_length × msg_repeat``
    """

    def __init__(
        self,
        message_length:       int   = 32,
        key:                  int   = 42,
        alpha:                float = 75.0,
        msg_repeat:           int   = 35,
        neighborhood_size:    int   = 5,
        robust_to_transforms: bool  = True,
        confidence_threshold: float = 0.90,
    ) -> None:
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")

        self.message_length       = message_length
        self.key                  = key
        self.alpha                = float(alpha)
        self.msg_repeat           = msg_repeat
        self.neighborhood_size    = neighborhood_size
        self.robust_to_transforms = robust_to_transforms
        self.confidence_threshold = confidence_threshold

        self._total_embedded: int = message_length * msg_repeat

        # ── Neighbourhood ring offsets — precomputed once ─────────────────
        # All (Δrow, Δcol) pairs in the k×k window except the center (0, 0).
        half   = neighborhood_size // 2
        dr, dc = np.mgrid[-half : half + 1, -half : half + 1]
        dr, dc = dr.ravel(), dc.ravel()
        ring   = (dr != 0) | (dc != 0)
        self._DR:  np.ndarray = dr[ring].astype(np.intp)   # (k²−1,)
        self._DC:  np.ndarray = dc[ring].astype(np.intp)   # (k²−1,)
        self._pad: int        = half

    # ── Internal helpers ──────────────────────────────────────────────────

    def _pixel_indices(
        self, H: int, W: int, C: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Draw ``_total_embedded`` unique pixel positions from H×W×C using
        a seeded RNG — identical seed → identical positions every call.

        Callers that have applied a geometric transform must pass the
        *post-transform* (H, W).  For lossless ±90° rotations H and W
        simply swap, so the same seeded draw addresses the correct pixels
        in the transformed image — no special handling required.

        Raises
        ------
        ValueError if the image is too small to host the payload.
        """
        n   = self._total_embedded
        cap = H * W * C
        if n > cap:
            raise ValueError(
                f"Image too small: need {n} pixel positions, "
                f"capacity is {cap} ({H}×{W}×{C})."
            )
        flat = np.random.default_rng(self.key).choice(cap, size=n, replace=False)
        return np.unravel_index(flat, (H, W, C))   # type: ignore[return-value]

    def _confidence_scores(
        self,
        img:   np.ndarray,
        rows:  np.ndarray,
        cols:  np.ndarray,
        chans: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a real-valued score at each embedded pixel position::

            score_i = img[row_i, col_i, chan_i]
                    − mean( k×k neighbourhood, center excluded )

        Reflect-padding (width = ``_pad``) handles border pixels without
        introducing artificial edges.  Sign encodes the decoded bit;
        magnitude encodes confidence.  Both metrics are derived from these.

        Returns
        -------
        np.ndarray, shape (_total_embedded,), float32
        """
        padded = np.pad(
            img,
            ((self._pad, self._pad), (self._pad, self._pad), (0, 0)),
            mode="reflect",
        )
        nr  = rows[:, None] + self._pad + self._DR          # (n, k²−1)
        nc  = cols[:, None] + self._pad + self._DC
        nch = np.broadcast_to(chans[:, None], nr.shape).astype(np.intp)

        return img[rows, cols, chans] - padded[nr, nc, nch].mean(axis=1)

    def _decode_once(
        self, img: np.ndarray, transform: str = "identity"
    ) -> DecodeResult:
        """
        Decode the watermark from a single (H, W, C) float32 image and
        compute both quality metrics.

        Pixel indices are derived from ``img.shape`` — caller is responsible
        for passing the already-transformed image.

        Parameters
        ----------
        img       : np.ndarray, shape (H, W, C), float32
        transform : str — echoed into the returned DecodeResult

        Returns
        -------
        DecodeResult
        """
        H, W, C = img.shape
        rows, cols, chans = self._pixel_indices(H, W, C)
        scores = self._confidence_scores(img, rows, cols, chans)

        # ── Metric 1: embedding strength ──────────────────────────────────
        # mean |score| over all selected pixels.
        # ≈ alpha if those pixels were actually shifted by ±alpha at encode time.
        embedding_strength = float(np.abs(scores).mean())

        # ── Metric 2: per-bit copy agreement → overall confidence ─────────
        # Reshape raw scores into (msg_repeat, message_length): one row per copy.
        scores_2d = scores.reshape(self.msg_repeat, self.message_length)

        # Hard bit decision for every (copy, bit-position) pair.
        bits_2d  = scores_2d >= 0.0                          # (msg_repeat, msg_len)

        # Fraction of copies voting "1" at each bit position.
        frac_one = bits_2d.mean(axis=0)                      # (message_length,)

        # Per-bit confidence = how strongly copies agree with majority.
        # max(p, 1−p) is 1.0 when all copies agree, 0.5 when perfectly split.
        bit_conf = np.maximum(frac_one, 1.0 - frac_one)      # (message_length,)
        overall_confidence = float(bit_conf.mean())

        # Final bits: majority vote (equivalent to rounding frac_one).
        bits = (frac_one >= 0.5).astype(np.uint8)

        return DecodeResult(
            bits=bits,
            embedding_strength=embedding_strength,
            overall_confidence=overall_confidence,
            transform=transform,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """
        Embed *watermark* into *image* via additive pixel modulation.

        Each selected pixel is shifted by::

            Δ = alpha × (2 × bit − 1)     # +alpha for bit=1, −alpha for bit=0

        Payload ``= tile(watermark, msg_repeat)`` — all copies use the same
        seeded pixel positions so they can be averaged at decode time.
        Output is clipped to [0, 255] and cast to the original dtype.

        Parameters
        ----------
        image     : np.ndarray, shape (H, W) or (H, W, C), uint8 or float
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

        payload                   = np.tile(np.asarray(watermark, dtype=np.uint8), self.msg_repeat)
        rows, cols, chans         = self._pixel_indices(H, W, C)
        img[rows, cols, chans]   += self.alpha * (2.0 * payload.astype(np.float32) - 1.0)
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the watermark from *image*; return bits only.

        Thin wrapper around :meth:`decode_verbose` — use that method to
        also obtain quality metrics and the winning transform name.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
        """
        return self.decode_verbose(image).bits

    def decode_verbose(self, image: np.ndarray) -> DecodeResult:
        """
        Extract the watermark from *image* with full diagnostic information.

        Without geometric robustness (``robust_to_transforms=False``)
            Single decode pass on the image as-is.

        With geometric robustness (``robust_to_transforms=True``)
            For each candidate in ``_CANDIDATE_TRANSFORMS`` (cheapest first):

            1. Apply the candidate inverse transform — H and W swap for
               rot90 / rot270, which is handled automatically by
               ``_pixel_indices`` using the post-transform shape.
            2. Compute ``overall_confidence`` (primary ranking criterion).
               This metric is insensitive to bit values and converges to 0.5
               for wrong orientations regardless of image content.
            3. **Early exit** if ``overall_confidence ≥ confidence_threshold``.
            4. After all candidates, return the highest-confidence result.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        DecodeResult — bits, embedding_strength, overall_confidence, transform
        """
        grayscale = image.ndim == 2
        img       = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]

        # ── Fast path: no transform search ────────────────────────────────
        if not self.robust_to_transforms:
            return self._decode_once(img, transform="identity")

        # ── Robust path: search over lossless inverse transforms ──────────
        best: Optional[DecodeResult] = None

        for name in _CANDIDATE_TRANSFORMS:
            transformed = _apply_transform(img, name)

            try:
                result = self._decode_once(transformed, transform=name)
            except ValueError:
                continue  # transformed image too small (rare edge case)

            if best is None or result.overall_confidence > best.overall_confidence:
                best = result

            if result.overall_confidence >= self.confidence_threshold:
                break  # reliable decode found — skip remaining candidates

        assert best is not None, "All candidate transforms failed (image too small?)."
        return best
