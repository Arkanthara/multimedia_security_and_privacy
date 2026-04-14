"""
model.py
========
Vectorized blind watermarking — confidence-weighted decoding,
optional robustness to lossless geometric transforms (flips, 90° rotations),
and optional FFT-based rotation-robust decoding for arbitrary rotation angles.

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

FFT rotation-search strategy
------------------------------
When ``enable_rotation_search=True`` and all lossless transforms fail, the
decoder switches to a three-phase FFT-guided approach:

**Phase 1 — FFT angle estimation** (:func:`estimate_rotation_fft`):
    The 2D FFT log-magnitude spectrum is computed and its central DC region
    suppressed.  A fast, fully-vectorised angular sweep accumulates spectral
    energy per 1°-wide bin over [0°, 180°).  The peak bin identifies the
    dominant frequency-domain direction, folded to [0°, 90°) to account for
    the centro-symmetry of the FFT magnitude.

**Phase 2 — Rotation + lossless search**:
    The image is rotated by the negative of the estimated angle (corrective
    inverse rotation) and all six lossless transforms are re-tried.
    Because the FFT angle is only known modulo 90°, combining with the full
    lossless catalogue covers all four 90°-multiple ambiguities.

**Phase 3 — Local refinement** (:meth:`WatermarkModel._refine_angle_search`):
    A two-level grid search (coarse → fine) explores a configurable
    neighbourhood of the FFT estimate.  At each test angle, all lossless
    transforms are applied.  Early-exit fires as soon as
    ``overall_confidence ≥ confidence_threshold``.

Dependencies
------------
numpy, opencv-python (cv2 re-exported for callers; also used internally for
bilinear-interpolated rotation).
"""

from __future__ import annotations

from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import cv2          # noqa: F401 — re-exported for callers (e.g. PSNR checks)
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Default rotation-search hyper-parameters
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_ROTATION_PARAMS: Dict[str, Any] = {
    # FFT angle estimation
    "n_angles":        180,   # angular bins spanning [0°, 180°); higher → finer
    "suppress_radius": 0.05,  # DC suppression radius as fraction of min(H,W)/2

    # Coarse local search around the FFT estimate
    "coarse_step":     1.0,   # step size in degrees
    "coarse_range":    2.0,   # ± search window in degrees around FFT estimate

    # Fine local search centred on the best coarse angle
    "fine_step":       0.5,   # step size in degrees
    "fine_range":      1.0,   # ± search window in degrees around best coarse hit
}


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


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate *img* counter-clockwise by *angle_deg* degrees in-place shape.

    Bilinear interpolation preserves the watermark signal embedded in pixel
    values.  Border pixels are filled by reflection (``cv2.BORDER_REFLECT``)
    to avoid introducing black corners that could corrupt watermark positions
    near the image boundary.

    Parameters
    ----------
    img       : np.ndarray, shape (H, W, C), float32
    angle_deg : float — counter-clockwise rotation angle in degrees

    Returns
    -------
    np.ndarray, shape (H, W, C), float32 — rotated image (same spatial size)
    """
    H, W = img.shape[:2]
    cx, cy = W * 0.5, H * 0.5
    M = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# FFT-based rotation estimation  (module-level, reusable)
# ─────────────────────────────────────────────────────────────────────────────

def estimate_rotation_fft(
    image: np.ndarray,
    n_angles: int = 180,
    suppress_radius: float = 0.05,
) -> float:
    """
    Estimate the dominant rotation angle of an image using FFT angular analysis.

    The 2D FFT magnitude spectrum encodes dominant orientations: spatial
    structures at angle θ produce spectral energy at θ + 90°.  By sweeping
    radial directions through the log-magnitude spectrum and accumulating
    energy per angular bin, the peak bin identifies the dominant frequency-
    domain direction, which is then mapped back to the spatial rotation angle.

    The FFT magnitude is centro-symmetric (|F(−u,−v)| = |F(u,v)|), so the
    full [0°, 180°) range is analysed and the result is folded modulo 90°.
    Callers must therefore combine the returned angle with all four
    lossless 90°-multiple transforms to resolve the quadrant ambiguity.

    Implementation notes
    --------------------
    * Fully vectorised — no Python loop over pixels or angles.
    * Runs in O(H·W + N) time after the O(H·W·log(H·W)) FFT.

    Parameters
    ----------
    image : np.ndarray, shape (H, W) or (H, W, C)
        Input image (uint8 or float32).  Converted to grayscale internally.
    n_angles : int, default 180
        Number of angular bins spanning [0°, 180°).  Higher values yield
        finer angular resolution at negligible extra cost (bincount is O(n)).
    suppress_radius : float, default 0.05
        Fraction of ``min(H, W) / 2`` zeroed out around the DC component.
        Suppressing the bright central peak prevents DC dominance from
        biasing the energy accumulation toward the centre angle.

    Returns
    -------
    float
        Estimated dominant angle in **[0°, 90°)**, degrees (modulo 90°).
    """
    # ── 1. Convert to grayscale float32 ──────────────────────────────────────
    if image.ndim == 3:
        gray = image.astype(np.float32).mean(axis=2)
    else:
        gray = image.astype(np.float32)

    H, W = gray.shape

    # ── 2. FFT → centre-shift → log magnitude ────────────────────────────────
    fft_mag = np.abs(np.fft.fftshift(np.fft.fft2(gray)))
    log_mag = np.log1p(fft_mag)                          # log1p avoids log(0)

    # ── 3. Suppress central DC region ────────────────────────────────────────
    cy, cx  = H // 2, W // 2
    Y, X    = np.mgrid[0:H, 0:W]
    r       = np.hypot(X - cx, Y - cy)
    r_limit = suppress_radius * (min(H, W) / 2.0)
    log_mag[r < r_limit] = 0.0

    # ── 4. Angular energy accumulation (fully vectorised) ────────────────────
    # Each pixel's angle from the centre, mapped to [0°, 180°) by the mod-π
    # trick: arctan2 ∈ (−π, π], so (arctan2 % π) ∈ [0, π).
    dx         = (X - cx).ravel().astype(np.float64)
    dy         = (Y - cy).ravel().astype(np.float64)
    angles_deg = np.degrees(np.arctan2(dy, dx) % np.pi)  # [0°, 180°)

    # Fast digitise into n_angles uniform bins.
    bin_edges  = np.linspace(0.0, 180.0, n_angles + 1)
    bin_idx    = np.searchsorted(bin_edges[1:], angles_deg)   # O(n log n)
    bin_idx    = np.clip(bin_idx, 0, n_angles - 1)

    # Weighted sum of log-magnitude per angular bin.
    energy = np.bincount(bin_idx, weights=log_mag.ravel(), minlength=n_angles)

    # ── 5. Peak angle, folded to [0°, 90°) via FFT symmetry ─────────────────
    peak_angle  = (float(np.argmax(energy)) / n_angles) * 180.0   # [0°, 180°)
    angle_mod90 = peak_angle % 90.0                                # [0°, 90°)

    return angle_mod90


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
        For rotation-corrected results the format is
        ``"rot_{angle:.2f}deg+{lossless_name}"``.
    angle_deg : float, default 0.0
        Corrective rotation angle applied before lossless transforms (degrees).
        Zero for purely lossless results; non-zero for FFT-guided ones.
    """
    bits:               np.ndarray
    embedding_strength: float
    overall_confidence: float
    transform:          str
    angle_deg:          float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WatermarkModel
# ─────────────────────────────────────────────────────────────────────────────

class WatermarkModel:
    """
    Vectorized blind watermarking — confidence-weighted decoding,
    optionally robust to lossless geometric transforms and arbitrary rotations.

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
        When True, :meth:`decode_verbose` searches over all lossless candidate
        inverse transforms and returns the highest-confidence result.
    confidence_threshold : float, default 0.90
        Early-exit threshold for all transform / rotation searches.
        A result with ``overall_confidence ≥ confidence_threshold`` is
        accepted immediately (range (0.5, 1.0]; 0.90 → 90 % copy agreement).
    enable_rotation_search : bool, default False
        When True (and ``robust_to_transforms=True``), activates the FFT-based
        rotation-robust decoding pipeline after the lossless-transform search
        fails.  See module docstring for the full three-phase strategy.
    rotation_search_params : dict, optional
        Override any subset of ``DEFAULT_ROTATION_PARAMS``.  Unknown keys are
        silently ignored.  Example::

            {"coarse_step": 0.5, "fine_step": 0.25}

    Total embedded bits
    -------------------
    ``message_length × msg_repeat``
    """

    def __init__(
        self,
        message_length:         int            = 32,
        key:                    int            = 42,
        alpha:                  float          = 75.0,
        msg_repeat:             int            = 35,
        neighborhood_size:      int            = 5,
        robust_to_transforms:   bool           = True,
        confidence_threshold:   float          = 0.90,
        enable_rotation_search: bool           = False,
        rotation_search_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")

        self.message_length          = message_length
        self.key                     = key
        self.alpha                   = float(alpha)
        self.msg_repeat              = msg_repeat
        self.neighborhood_size       = neighborhood_size
        self.robust_to_transforms    = robust_to_transforms
        self.confidence_threshold    = confidence_threshold
        self.enable_rotation_search  = enable_rotation_search
        self._rotation_search_params: Dict[str, Any] = rotation_search_params or {}

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

    # ── Internal helpers ──────────────────────────────────────────────────────

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

        Parameters
        ----------
        H, W, C : int — image dimensions

        Returns
        -------
        rows, cols, chans : np.ndarray — each shape (_total_embedded,)

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

        Parameters
        ----------
        img   : np.ndarray, shape (H, W, C), float32
        rows  : np.ndarray, shape (_total_embedded,), intp
        cols  : np.ndarray, shape (_total_embedded,), intp
        chans : np.ndarray, shape (_total_embedded,), intp

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
        self,
        img:       np.ndarray,
        transform: str   = "identity",
        angle_deg: float = 0.0,
    ) -> DecodeResult:
        """
        Decode the watermark from a single (H, W, C) float32 image and
        compute both quality metrics.

        Pixel indices are derived from ``img.shape`` — caller is responsible
        for passing the already-transformed image.

        Parameters
        ----------
        img       : np.ndarray, shape (H, W, C), float32
        transform : str — echoed into the returned :class:`DecodeResult`
        angle_deg : float — corrective rotation angle echoed into the result

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
            angle_deg=angle_deg,
        )

    def _try_lossless_transforms(
        self,
        img:              np.ndarray,
        transform_prefix: str   = "",
        angle_deg:        float = 0.0,
    ) -> Tuple[Optional[DecodeResult], bool]:
        """
        Attempt decoding under all lossless inverse transforms and return
        the best result together with a success flag.

        Iterates through ``_CANDIDATE_TRANSFORMS`` in cheapest-first order.
        Applies early exit as soon as a result meets ``confidence_threshold``.

        Parameters
        ----------
        img              : np.ndarray, shape (H, W, C), float32
        transform_prefix : str — prepended to each transform name in the label,
                           e.g. ``"fft_rot_23.40deg"`` → label becomes
                           ``"fft_rot_23.40deg+identity"``.
        angle_deg        : float — corrective rotation angle recorded in the result

        Returns
        -------
        best    : Optional[DecodeResult] — highest-confidence result seen so far
                  (``None`` only if every transform raised ValueError)
        success : bool — ``True`` if ``confidence_threshold`` was reached
        """
        best: Optional[DecodeResult] = None

        for name in _CANDIDATE_TRANSFORMS:
            transformed = _apply_transform(img, name)
            label = f"{transform_prefix}+{name}" if transform_prefix else name

            try:
                result = self._decode_once(transformed, transform=label,
                                           angle_deg=angle_deg)
            except ValueError:
                continue   # image too small after transform (rare edge case)

            if best is None or result.overall_confidence > best.overall_confidence:
                best = result

            if result.overall_confidence >= self.confidence_threshold:
                return best, True   # early exit — reliable decode found

        return best, False

    def _refine_angle_search(
        self,
        img:        np.ndarray,
        base_angle: float,
        params:     Dict[str, Any],
    ) -> Tuple[Optional[DecodeResult], bool]:
        """
        Multi-resolution local angular search around *base_angle*.

        Two search passes are performed in order:

        **Coarse pass**
            Angles in ``[base_angle − coarse_range, base_angle + coarse_range]``
            sampled every ``coarse_step`` degrees.

        **Fine pass**
            Angles in ``[best_coarse − fine_range, best_coarse + fine_range]``
            sampled every ``fine_step`` degrees, centred on the angle that gave
            the highest confidence in the coarse pass.

        After each candidate angle the image is rotated by ``−angle`` (corrective
        inverse rotation) and all six lossless transforms are attempted.  An
        early exit is triggered as soon as ``confidence_threshold`` is met.

        Parameters
        ----------
        img        : np.ndarray, shape (H, W, C), float32
            Original (pre-rotation) image.
        base_angle : float
            FFT-estimated dominant angle in degrees; centre of the search grid.
        params     : dict
            Merged rotation-search parameters (see ``DEFAULT_ROTATION_PARAMS``).

        Returns
        -------
        best    : Optional[DecodeResult]
        success : bool — ``True`` if ``confidence_threshold`` was reached
        """
        coarse_step  = float(params.get("coarse_step",  DEFAULT_ROTATION_PARAMS["coarse_step"]))
        coarse_range = float(params.get("coarse_range", DEFAULT_ROTATION_PARAMS["coarse_range"]))
        fine_step    = float(params.get("fine_step",    DEFAULT_ROTATION_PARAMS["fine_step"]))
        fine_range   = float(params.get("fine_range",   DEFAULT_ROTATION_PARAMS["fine_range"]))

        best: Optional[DecodeResult] = None
        best_coarse_angle = base_angle

        # ── Coarse pass ───────────────────────────────────────────────────────
        coarse_angles = np.arange(
            base_angle - coarse_range,
            base_angle + coarse_range + coarse_step * 0.5,   # inclusive endpoint
            coarse_step,
        )
        for angle in coarse_angles:
            rotated   = _rotate_image(img, -float(angle))
            candidate, success = self._try_lossless_transforms(
                rotated,
                transform_prefix=f"refine_rot_{angle:.2f}deg",
                angle_deg=float(angle),
            )
            if candidate is not None:
                if best is None or candidate.overall_confidence > best.overall_confidence:
                    best = candidate
                    best_coarse_angle = float(angle)
            if success:
                return best, True

        # ── Fine pass — centred on the best coarse angle ──────────────────────
        fine_angles = np.arange(
            best_coarse_angle - fine_range,
            best_coarse_angle + fine_range + fine_step * 0.5,
            fine_step,
        )
        for angle in fine_angles:
            rotated   = _rotate_image(img, -float(angle))
            candidate, success = self._try_lossless_transforms(
                rotated,
                transform_prefix=f"fine_rot_{angle:.2f}deg",
                angle_deg=float(angle),
            )
            if candidate is not None:
                if best is None or candidate.overall_confidence > best.overall_confidence:
                    best = candidate
            if success:
                return best, True

        return best, False

    # ── Public API ────────────────────────────────────────────────────────────

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
        also obtain quality metrics, the winning transform name, and the
        applied corrective rotation angle.

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

        The pipeline executes in strictly ordered phases, with an **early exit**
        after each phase if ``overall_confidence ≥ confidence_threshold``.

        **Phase 0 — Fast path** (``robust_to_transforms=False``)
            Single decode pass on the image as-is.  Returns immediately.

        **Phase 1 — Baseline decode**
            Attempts decoding on the unmodified image (identity transform).

        **Phase 2 — Lossless transform search**
            Tries the remaining five lossless inverse transforms in cheapest-
            first order: hflip → vflip → rot270 → rot90 → rot180.

        *Phases 3–5 are active only when* ``enable_rotation_search=True``.

        **Phase 3 — FFT rotation estimation + corrective rotation**
            Calls :func:`estimate_rotation_fft` to obtain the dominant angle
            (modulo 90°, due to FFT centro-symmetry).  Rotates the image by
            the negative of that angle and re-runs all six lossless transforms.
            The quadrant ambiguity is automatically resolved by the lossless
            catalogue covering all four 90°-multiples.

        **Phase 4 — Local angular refinement**
            Performs a two-level (coarse → fine) grid search around the FFT
            estimate via :meth:`_refine_angle_search`.  At each candidate
            angle, all six lossless transforms are tried.

        **Phase 5 — Fallback**
            Returns the highest-confidence :class:`DecodeResult` accumulated
            across all phases, even if no phase reached the threshold.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        DecodeResult
            ``bits``, ``embedding_strength``, ``overall_confidence``,
            ``transform``, ``angle_deg``.
        """
        grayscale = image.ndim == 2
        img       = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]

        # ── Phase 0: fast path — no transform search ──────────────────────────
        if not self.robust_to_transforms:
            return self._decode_once(img, transform="identity")

        # ── Phase 1: baseline decode (identity, no rotation) ─────────────────
        best: Optional[DecodeResult] = None
        try:
            best = self._decode_once(img, transform="identity")
        except ValueError:
            pass

        if best is not None and best.overall_confidence >= self.confidence_threshold:
            return best

        # ── Phase 2: lossless transform search ───────────────────────────────
        # Skip identity (index 0) — already attempted in Phase 1.
        for name in _CANDIDATE_TRANSFORMS[1:]:
            transformed = _apply_transform(img, name)
            try:
                result = self._decode_once(transformed, transform=name)
            except ValueError:
                continue

            if best is None or result.overall_confidence > best.overall_confidence:
                best = result

            if result.overall_confidence >= self.confidence_threshold:
                return best   # early exit

        # ── Phases 3–5: FFT-guided rotation search ────────────────────────────
        if not self.enable_rotation_search:
            # Rotation search disabled — return best lossless result.
            assert best is not None, \
                "All lossless transforms failed (image too small?)."
            return best

        # Merge caller overrides with library defaults.
        params: Dict[str, Any] = {**DEFAULT_ROTATION_PARAMS,
                                   **self._rotation_search_params}

        # ── Phase 3: FFT angle estimation → corrective rotation ───────────────
        angle_est = estimate_rotation_fft(
            img,
            n_angles=int(params["n_angles"]),
            suppress_radius=float(params["suppress_radius"]),
        )

        # Rotate by −angle_est (inverse of the estimated rotation) and try
        # all six lossless transforms.  This resolves the 90°-quadrant
        # ambiguity without any additional brute-force cost.
        rotated_fft = _rotate_image(img, -angle_est)
        candidate, success = self._try_lossless_transforms(
            rotated_fft,
            transform_prefix=f"fft_rot_{angle_est:.2f}deg",
            angle_deg=angle_est,
        )
        if candidate is not None:
            if best is None or candidate.overall_confidence > best.overall_confidence:
                best = candidate
        if success:
            return best   # type: ignore[return-value]

        # ── Phase 4: local angular refinement ────────────────────────────────
        candidate, success = self._refine_angle_search(img, angle_est, params)
        if candidate is not None:
            if best is None or candidate.overall_confidence > best.overall_confidence:
                best = candidate
        if success:
            return best   # type: ignore[return-value]

        # ── Phase 5: fallback — best candidate across all phases ──────────────
        assert best is not None, \
            "All decode attempts failed (image too small?)."
        return best
