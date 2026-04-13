"""
model.py
========
Vectorized blind watermarking — confidence-weighted decoding,
optional robustness to lossless geometric transforms (flips, 90° rotations)
and arbitrary rotation via Fourier-domain angle estimation.

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

Phase 1 — Lossless transform search
------------------------------------
When ``robust_to_transforms=True``, the decoder tries six lossless candidate
inverse transforms ordered cheapest-first::

    identity → hflip → vflip → rot270 → rot90 → rot180

All transforms are lossless (numpy views or ``numpy.rot90``): no interpolation,
no resolution change, and — for ±90° rotations — H and W simply swap so
``_pixel_indices`` adapts automatically.

Phase 2 — Fourier-based arbitrary rotation correction
------------------------------------------------------
When ``robust_to_rotation=True``, the decoder additionally estimates the image
rotation angle from its Fourier magnitude spectrum and applies a corrective
rotation before extraction.  This phase fires only if Phase 1 did not already
exceed ``confidence_threshold``.

Physical principle
~~~~~~~~~~~~~~~~~~
The Fourier transform of a rotated image is rotated by the same angle.
Natural images typically contain strong horizontal structures (scan lines,
horizons, text baselines), whose energy in the frequency domain concentrates
near the vertical axis (≈ 90°).  If the image has been rotated by φ, this
energy peak shifts to approximately 90° + φ.  Subtracting 90° therefore
recovers an estimate of φ, modulo 180° (Hermitian symmetry of real-valued
FFTs collapses the full circle to a half-circle).

Algorithm
~~~~~~~~~
1. Convert the watermarked image to grayscale; compute its 2-D FFT.
2. Build a log-compressed, DC-centred magnitude spectrum.
3. For each pixel in an annular validity region, record its angular position
   θ ∈ [0°, 180°) and accumulate its log-magnitude into an angular histogram.
4. Smooth the histogram with a Hanning window and find its peak θ_peak.
5. ``θ_est = θ_peak − 90°`` gives the estimated spatial rotation (−90°, 90°).
6. Build both sign hypotheses and try 90° offsets for each:

    ``(−θ_est + k × 90°) mod 360°`` and ``(+θ_est + k × 90°) mod 360°``

    for k ∈ {0, 1, 2, 3}. This resolves Hermitian 180° ambiguity, 90° quadrant
    ambiguity, and estimator sign ambiguity.
7. For each base candidate, run a coarse local angular sweep (FFT-guided).
8. Around the best coarse angles, run a fine local sweep for sub-degree
    precision.
9. For the top refined angles, search over lossless transforms (including
    flips and 90° rotations) to handle wrong-sense inverse-rotation cases.

For each candidate (both phases)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  1. Apply the inverse-transform / corrective-rotation candidate.
  2. Rank by ``overall_confidence`` (primary criterion).
  3. **Early exit** if ``overall_confidence ≥ confidence_threshold``.
  4. After all candidates, return the highest-confidence result.

This pipeline makes the decoder simultaneously robust to:
  * Horizontal and vertical flip
  * JPEG compression and Gaussian blur
  * Exact 90° / 180° / 270° rotations (lossless, Phase 1)
  * Arbitrary rotation angles (Fourier-corrected, Phase 2)

Dependencies
------------
numpy, opencv-python (cv2 — used internally for bicubic rotation in Phase 2;
also re-exported for callers performing PSNR checks, etc.).
"""

from __future__ import annotations

from typing import Dict, FrozenSet, List, NamedTuple, Optional, Tuple

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

# Rotation angles (mod 360°) already covered by the lossless Phase-1 search.
# Phase 2 skips these to avoid redundant, lower-quality (interpolated) attempts.
_LOSSLESS_ANGLES: FrozenSet[float] = frozenset({0.0, 90.0, 180.0, 270.0})


def _angular_distance_deg(a: float, b: float) -> float:
    """
    Return the smallest absolute angular distance between *a* and *b*.

    Parameters
    ----------
    a, b : float
        Angles in degrees, interpreted modulo 360°.

    Returns
    -------
    float
        Value in [0°, 180°].
    """
    return float(abs(((a - b + 180.0) % 360.0) - 180.0))


def _append_unique_angle(
    angles: List[float],
    angle: float,
    tol: float = 0.25,
) -> None:
    """
    Append *angle* to *angles* if no existing entry is within *tol* degrees.

    Parameters
    ----------
    angles : list[float]
        Mutable list of angles (degrees).
    angle : float
        Candidate angle to append.
    tol : float, default 0.25
        Circular-distance tolerance used for deduplication.
    """
    if not any(_angular_distance_deg(angle, a) <= tol for a in angles):
        angles.append(float(angle))


# ─────────────────────────────────────────────────────────────────────────────
# Module-level geometric helpers
# ─────────────────────────────────────────────────────────────────────────────

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
    if name == "rot90":                       # 90° CCW
        return np.rot90(img, k=1, axes=(0, 1))
    if name == "rot180":
        return np.rot90(img, k=2, axes=(0, 1))
    if name == "rot270":                      # 90° CW
        return np.rot90(img, k=3, axes=(0, 1))
    raise ValueError(f"Unknown transform: {name!r}")


def _fourier_rotation_estimate(
    img_gray: np.ndarray,
    n_angles: int = 360,
) -> float:
    """
    Estimate the spatial rotation angle of an image from its Fourier magnitude
    spectrum.

    **Physical principle**

    The Fourier transform of a rotated image is rotated by the same angle.
    Natural images typically contain strong horizontal structures (scan lines,
    horizons, text baselines) whose energy in the frequency domain concentrates
    near the vertical axis (≈ 90°).  If the image has been rotated by φ, this
    energy peak shifts to approximately 90° + φ.  Subtracting 90° therefore
    recovers an estimate of φ, modulo 180° (Hermitian symmetry of real FFTs
    collapses the full circle onto a half-circle).

    **Algorithm**

    1. Compute the 2-D FFT magnitude spectrum, log-compressed and DC-centred.
    2. Build polar coordinates (r, θ) relative to the spectrum centre.
    3. Accumulate log-magnitude into an angular histogram over [0°, 180°),
       restricted to an annular region that excludes the DC blob and the
       Nyquist boundary.
    4. Smooth the histogram with a Hanning window to suppress bin-level noise.
    5. Find the peak bin and subtract 90° to convert the frequency-domain
       peak angle into a spatial rotation estimate.

    Parameters
    ----------
    img_gray : np.ndarray, shape (H, W), float32
        Single-channel (grayscale) version of the image.
    n_angles : int, default 360
        Number of angular bins covering [0°, 180°).  Higher values give a
        finer estimate at modest extra cost (FFT cost is fixed; only the
        histogram grows).

    Returns
    -------
    float
        Estimated spatial rotation angle in degrees, in the range [−90°, 90°).
        This is the angle by which the image appears to have been rotated.
        The caller should try both sign hypotheses (``−estimate`` and
        ``+estimate``), each with 90° offsets, to resolve Hermitian, quadrant,
        and sign ambiguities.

    Notes
    -----
    The estimate is reliable for images with a clear dominant orientation
    (natural scenes, documents).  For synthetic or texture-less images the
    confidence-based selection in :meth:`WatermarkModel.decode_verbose` will
    automatically favour a better candidate from Phase 1.
    """
    H, W = img_gray.shape

    # ── 1. Log-compressed, DC-centred magnitude spectrum ──────────────────
    magnitude = np.abs(np.fft.fftshift(np.fft.fft2(img_gray)))  # (H, W)
    log_mag   = np.log1p(magnitude)                              # compress dynamic range

    # ── 2. Polar coordinates relative to spectrum centre ──────────────────
    cy, cx = H // 2, W // 2
    y  = (np.arange(H, dtype=np.float32) - cy)[:, np.newaxis]   # (H, 1)
    x  = (np.arange(W, dtype=np.float32) - cx)[np.newaxis, :]   # (1, W)

    # Broadcasting produces (H, W) arrays without explicit meshgrid.
    r     = np.hypot(y, x)                                        # (H, W)
    theta = np.degrees(np.arctan2(y, x)) % 180.0                 # (H, W) ∈ [0°, 180°)

    # ── 3. Annular validity mask — exclude DC blob and Nyquist boundary ───
    # r_min: 2 % of the longest dimension keeps DC and its ringing out.
    # r_max: 45 % of the shortest dimension avoids wrap-around artefacts.
    r_min = max(H, W) * 0.02
    r_max = min(H, W) * 0.45
    mask  = (r >= r_min) & (r <= r_max)                           # (H, W) bool

    # ── 4. Angular energy histogram (vectorized; no Python loop) ──────────
    bins    = np.linspace(0.0, 180.0, n_angles + 1)
    hist, _ = np.histogram(theta[mask], bins=bins, weights=log_mag[mask])

    # ── 5. Hanning-window smoothing — suppresses single-bin noise ─────────
    win         = np.hanning(11).astype(np.float64)
    win        /= win.sum()
    hist_smooth = np.convolve(hist, win, mode="same")

    # ── 6. Peak angle → spatial rotation estimate ─────────────────────────
    peak_idx   = int(np.argmax(hist_smooth))
    peak_angle = 0.5 * (bins[peak_idx] + bins[peak_idx + 1])    # bin centre [0°, 180°)

    # For horizontal-dominant images the frequency-domain peak sits near 90°.
    # Subtracting 90° converts the frequency-domain angle into the estimated
    # spatial rotation, mapping the result to [−90°, 90°).
    return float(peak_angle - 90.0)


def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a float32 image by *angle_deg* degrees counter-clockwise around
    its centre, using bicubic interpolation and reflect border padding.

    Parameters
    ----------
    img       : np.ndarray, shape (H, W, C), float32
    angle_deg : float
        Rotation angle in degrees.  Positive values rotate counter-clockwise
        in standard (Cartesian) orientation — matching OpenCV's convention for
        ``cv2.getRotationMatrix2D`` when y increases downward in image space.

    Returns
    -------
    np.ndarray, shape (H, W, C), float32
        Rotated image at the same spatial resolution as the input.

    Notes
    -----
    ``cv2.INTER_CUBIC`` (bicubic) is preferred over bilinear for watermark
    recovery because it preserves pixel-level amplitude modulations more
    faithfully.  ``cv2.BORDER_REFLECT`` avoids the black-corner artefacts
    that constant-value padding would introduce near image edges, which would
    otherwise suppress watermark confidence in those regions.
    """
    H, W   = img.shape[:2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0
    M      = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (W, H),
        flags      = cv2.INTER_CUBIC,
        borderMode = cv2.BORDER_REFLECT,
    )


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
        Name of the lossless candidate transform applied *after* any corrective
        rotation (e.g. ``"identity"``, ``"hflip"``).
    rotation_angle : float
        Corrective counter-clockwise rotation (degrees) applied *before* the
        lossless transform.  0.0 for results produced by Phase 1 alone.
    """
    bits               : np.ndarray
    embedding_strength : float
    overall_confidence : float
    transform          : str
    rotation_angle     : float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# WatermarkModel
# ─────────────────────────────────────────────────────────────────────────────

class WatermarkModel:
    """
    Vectorized blind watermarking — confidence-weighted decoding, optionally
    robust to lossless geometric transforms and arbitrary rotation.

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
        When True, :meth:`decode_verbose` searches over all six lossless
        candidate inverse transforms (identity, hflip, vflip, rot90, rot180,
        rot270) in Phase 1 and returns the highest-confidence result.
    confidence_threshold : float, default 0.90
        Early-exit threshold applied in both search phases.
        A result with ``overall_confidence ≥ confidence_threshold`` is
        accepted immediately (range (0.5, 1.0]; 0.90 → 90 % copy agreement).
    robust_to_rotation : bool, default True
        When True, :meth:`decode_verbose` runs Phase 2: it estimates the
        image rotation angle from the Fourier magnitude spectrum and tries
        sign-robust corrective rotations using both ``−θ_est`` and ``+θ_est``
        with 90° offsets.  Each de-rotated image is then searched with the
        full lossless-transform set (including flips and 90° rotations) to
        handle wrong-sense inverse-rotation hypotheses.  Requires
        ``opencv-python`` for the bicubic warp.  Phase 2 fires only when
        Phase 1 did not already reach ``confidence_threshold``.
    rotation_fft_n_angles : int, default 360
        Angular resolution of the Fourier rotation estimator (number of bins
        covering [0°, 180°)).  Higher values give a finer estimate at modest
        additional cost; 180–360 is usually sufficient.

    Total embedded bits
    -------------------
    ``message_length × msg_repeat`` payload bits, plus a small fixed pilot
    set used internally for score-polarity disambiguation.
    """

    def __init__(
        self,
        message_length:        int   = 32,
        key:                   int   = 42,
        alpha:                 float = 75.0,
        msg_repeat:            int   = 35,
        neighborhood_size:     int   = 5,
        robust_to_transforms:  bool  = True,
        confidence_threshold:  float = 0.90,
        robust_to_rotation:    bool  = True,
        rotation_fft_n_angles: int   = 360,
    ) -> None:
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")

        self.message_length        = message_length
        self.key                   = key
        self.alpha                 = float(alpha)
        self.msg_repeat            = msg_repeat
        self.neighborhood_size     = neighborhood_size
        self.robust_to_transforms  = robust_to_transforms
        self.confidence_threshold  = confidence_threshold
        self.robust_to_rotation    = robust_to_rotation
        self.rotation_fft_n_angles = rotation_fft_n_angles

        self._total_embedded: int = message_length * msg_repeat
        # Pilot positions carry a known positive sign to disambiguate global
        # score inversion after heavy geometric/interpolation distortions.
        self._pilot_size: int = 64
        self._index_cache: Dict[
            Tuple[int, int, int],
            Tuple[np.ndarray, np.ndarray, np.ndarray],
        ] = {}

        # ── Neighbourhood ring offsets — precomputed once ─────────────────
        # All (Δrow, Δcol) pairs in the k×k window, excluding the centre (0,0).
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
        Draw deterministic unique pixel positions from H×W×C using a seeded
        RNG — identical seed → identical positions every call.

        The returned indices concatenate payload positions (length
        ``_total_embedded``) followed by pilot positions (length
        ``_pilot_size``).

        Callers that have applied a geometric transform must pass the
        *post-transform* (H, W).  For lossless ±90° rotations H and W
        simply swap, so the same seeded draw addresses the correct pixels
        in the transformed image — no special handling required.

        Raises
        ------
        ValueError
            If the image is too small to host payload and pilot positions.
        """
        shape = (H, W, C)
        cached = self._index_cache.get(shape)
        if cached is not None:
            return cached

        n   = self._total_embedded + self._pilot_size
        cap = H * W * C
        if n > cap:
            raise ValueError(
                f"Image too small: need {n} pixel positions, "
                f"capacity is {cap} ({H}×{W}×{C})."
            )
        flat = np.random.default_rng(self.key).choice(cap, size=n, replace=False)
        indices = np.unravel_index(flat, shape)   # type: ignore[assignment]
        self._index_cache[shape] = indices
        return indices

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
                    − mean( k×k neighbourhood, centre excluded )

        Reflect-padding (width = ``_pad``) handles border pixels without
        introducing artificial edges.  Sign encodes the decoded bit;
        magnitude encodes confidence.  Both quality metrics are derived from
        these scores.

        Parameters
        ----------
        img   : np.ndarray, shape (H, W, C), float32
        rows  : np.ndarray, shape (n,), int — row indices of embedded pixels
        cols  : np.ndarray, shape (n,), int — column indices
        chans : np.ndarray, shape (n,), int — channel indices

        Returns
        -------
        np.ndarray, shape (_total_embedded,), float32
        """
        padded = np.pad(
            img,
            ((self._pad, self._pad), (self._pad, self._pad), (0, 0)),
            mode="reflect",
        )
        # Neighbour coordinates: broadcast (n, 1) + (k²−1,) → (n, k²−1)
        nr  = rows[:, None] + self._pad + self._DR
        nc  = cols[:, None] + self._pad + self._DC
        nch = np.broadcast_to(chans[:, None], nr.shape).astype(np.intp)

        return img[rows, cols, chans] - padded[nr, nc, nch].mean(axis=1)

    def _decode_once(
        self,
        img:            np.ndarray,
        transform:      str   = "identity",
        rotation_angle: float = 0.0,
    ) -> DecodeResult:
        """
        Decode the watermark from a single (H, W, C) float32 image and
        compute both quality metrics.

        Pixel indices are derived from ``img.shape`` — the caller is
        responsible for passing the already-transformed image.

        Parameters
        ----------
        img            : np.ndarray, shape (H, W, C), float32
        transform      : str — lossless transform label, echoed in the result.
        rotation_angle : float — corrective rotation already applied (degrees
                         CCW); echoed in the result.

        Returns
        -------
        DecodeResult
        """
        H, W, C = img.shape
        rows, cols, chans = self._pixel_indices(H, W, C)
        scores_all        = self._confidence_scores(img, rows, cols, chans)

        payload_n = self._total_embedded
        scores    = scores_all[:payload_n]

        if self._pilot_size > 0:
            pilot_scores = scores_all[payload_n:]
            # Correct global sign inversion before bit decoding.
            if float(pilot_scores.mean()) < 0.0:
                scores = -scores

        # ── Metric 1: embedding strength ──────────────────────────────────
        # mean |score| over all selected pixels.
        # ≈ alpha when those pixels were shifted by ±alpha at encode time.
        embedding_strength = float(np.abs(scores).mean())

        # ── Metric 2: per-bit copy agreement → overall confidence ─────────
        # Reshape raw scores into (msg_repeat, message_length): one row per copy.
        scores_2d = scores.reshape(self.msg_repeat, self.message_length)

        # Hard bit decision for every (copy, bit-position) pair.
        bits_2d  = scores_2d >= 0.0                              # (msg_repeat, msg_len)

        # Fraction of copies voting "1" at each bit position.
        frac_one = bits_2d.mean(axis=0)                          # (message_length,)

        # Per-bit confidence = majority agreement strength.
        # max(p, 1−p) is 1.0 when all copies agree, 0.5 when perfectly split.
        bit_conf           = np.maximum(frac_one, 1.0 - frac_one)
        overall_confidence = float(bit_conf.mean())

        # Final bits: majority vote (equivalent to rounding frac_one).
        bits = (frac_one >= 0.5).astype(np.uint8)

        return DecodeResult(
            bits               = bits,
            embedding_strength = embedding_strength,
            overall_confidence = overall_confidence,
            transform          = transform,
            rotation_angle     = rotation_angle,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """
        Embed *watermark* into *image* via additive pixel modulation.

        Each selected pixel is shifted by::

            Δ = alpha × (2 × bit − 1)     # +alpha for bit=1, −alpha for bit=0

        Payload ``= tile(watermark, msg_repeat)`` — all copies use the same
        seeded pixel positions so they can be averaged at decode time.
        A small fixed set of pilot positions is additionally shifted by
        ``+alpha`` to resolve global score-sign ambiguity at decode time.
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

        payload                  = np.tile(np.asarray(watermark, dtype=np.uint8), self.msg_repeat)
        rows, cols, chans        = self._pixel_indices(H, W, C)
        payload_n                = self._total_embedded

        img[
            rows[:payload_n],
            cols[:payload_n],
            chans[:payload_n],
        ] += self.alpha * (2.0 * payload.astype(np.float32) - 1.0)

        # Embed known positive-sign pilot markers for polarity disambiguation.
        img[
            rows[payload_n:],
            cols[payload_n:],
            chans[payload_n:],
        ] += self.alpha
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the watermark from *image*; return bits only.

        Thin wrapper around :meth:`decode_verbose` — use that method to also
        obtain quality metrics, the winning transform name, and the corrective
        rotation angle.

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

        **Fast path** (``robust_to_transforms=False`` and
        ``robust_to_rotation=False``)
            Single decode pass on the image as-is.

        **Phase 1 — lossless transform search** (``robust_to_transforms=True``)
            For each candidate in ``_CANDIDATE_TRANSFORMS`` (cheapest first):

            1. Apply the candidate inverse transform (zero-copy view or
               ``numpy.rot90`` — no interpolation, no quality loss).
            2. Compute both quality metrics via :meth:`_decode_once`.
            3. **Early exit** if ``overall_confidence ≥ confidence_threshold``.
            4. Track the highest-confidence result seen so far.

            When ``robust_to_transforms=False``, only the identity transform
            is tried so that Phase 2 can still fire if needed.

          **Phase 2 — Fourier rotation correction** (``robust_to_rotation=True``,
          fires only when Phase 1 did not early-exit)
                1. Compute a grayscale version of the image and call
                    :func:`_fourier_rotation_estimate` to obtain ``θ_est``.
                2. Build sign-robust base candidates using both ``−θ_est`` and
                    ``+θ_est``, each with 90° offsets:

                    ``φ = (±θ_est + k × 90°) mod 360°`` for k ∈ {0, 1, 2, 3}.

                3. Expand each base candidate with a coarse local sweep
                    (FFT-guided angular refinement window).
                4. Score all coarse candidates with identity transform.
                5. Around the top-scoring coarse angles, run a finer local sweep
                    and rescore with identity for sub-degree precision.
                6. If confidence remains weak, run an adaptive dense local sweep
                    around all FFT base hypotheses.
                7. Run flip/90° disambiguation on only the best refined angles:

                    a. De-rotate by ``φ`` via :func:`_rotate_image`.
                    b. Evaluate ``hflip``, ``vflip``, ``rot90``, ``rot180``,
                        ``rot270``.
                    c. **Early exit** if ``overall_confidence ≥ confidence_threshold``.

        At the end of both phases, the result with the highest
        ``overall_confidence`` is returned.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        DecodeResult
            ``bits``, ``embedding_strength``, ``overall_confidence``,
            ``transform``, ``rotation_angle``.
        """
        grayscale = image.ndim == 2
        img       = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]

        # ── Fast path: no search at all ───────────────────────────────────
        if not self.robust_to_transforms and not self.robust_to_rotation:
            return self._decode_once(img)

        best: Optional[DecodeResult] = None

        # ── Phase 1: lossless transform search ───────────────────────────
        # When robust_to_rotation=True but robust_to_transforms=False we
        # still need to run the identity candidate so that `best` is
        # initialised before Phase 2 begins.
        lossless_names: List[str] = (
            _CANDIDATE_TRANSFORMS if self.robust_to_transforms else ["identity"]
        )
        for name in lossless_names:
            transformed = _apply_transform(img, name)
            try:
                result = self._decode_once(
                    transformed, transform=name, rotation_angle=0.0
                )
            except ValueError:
                continue   # image too small for this (swapped) shape — rare

            if best is None or result.overall_confidence > best.overall_confidence:
                best = result
            if result.overall_confidence >= self.confidence_threshold:
                return best  # type: ignore[return-value]

        # ── Phase 2: Fourier-based arbitrary rotation correction ──────────
        if self.robust_to_rotation:
            # Grayscale proxy for FFT (channel-mean; fully vectorized).
            img_gray  = img.mean(axis=2)                         # (H, W)
            theta_est = _fourier_rotation_estimate(
                img_gray, self.rotation_fft_n_angles
            )

            # Angles already covered losslessly in Phase 1 — skip to avoid
            # redundant, lower-quality (interpolated) repeat attempts.
            covered: FrozenSet[float] = (
                _LOSSLESS_ANGLES if self.robust_to_transforms else frozenset({0.0})
            )

            # Build both sign hypotheses (−θ and +θ), each with 90° offsets.
            offsets = np.array([0.0, 90.0, 180.0, 270.0], dtype=np.float64)
            raw_candidates = np.concatenate(
                [
                    (-theta_est + offsets) % 360.0,
                    (+theta_est + offsets) % 360.0,
                ]
            )

            base_angles: List[float] = []
            for phi in raw_candidates.tolist():
                # Skip angles effectively covered losslessly in Phase 1.
                if any(_angular_distance_deg(phi, a) < 0.1 for a in covered | {360.0}):
                    continue
                _append_unique_angle(base_angles, phi)

            # Coarse FFT-guided refinement around each base angle.
            coarse_offsets = np.array([-12.0, -8.0, -4.0, 0.0, 4.0, 8.0, 12.0])
            coarse_angles: List[float] = []
            for phi in base_angles:
                for dphi in coarse_offsets.tolist():
                    cand = (phi + dphi) % 360.0
                    if any(_angular_distance_deg(cand, a) < 0.1 for a in covered | {360.0}):
                        continue
                    _append_unique_angle(coarse_angles, cand, tol=0.1)

            if not coarse_angles:
                coarse_angles = list(base_angles)

            rotate_cache: Dict[float, np.ndarray] = {}
            identity_trials: List[Tuple[float, float]] = []

            def _eval_identity(phi: float) -> Optional[DecodeResult]:
                cached = rotate_cache.get(phi)
                if cached is None:
                    cached = _rotate_image(img, phi)
                    rotate_cache[phi] = cached
                try:
                    result_id = self._decode_once(
                        cached,
                        transform="identity",
                        rotation_angle=phi,
                    )
                except ValueError:
                    return None
                identity_trials.append((result_id.overall_confidence, phi))
                return result_id

            # First pass: evaluate coarse candidates with identity only.
            for phi in coarse_angles:
                result_id = _eval_identity(phi)
                if result_id is None:
                    continue
                if best is None or result_id.overall_confidence > best.overall_confidence:
                    best = result_id
                if result_id.overall_confidence >= self.confidence_threshold:
                    return best  # type: ignore[return-value]

            identity_trials.sort(key=lambda t: t[0], reverse=True)
            seed_angles = [phi for _, phi in identity_trials[:6]]

            # Second pass: sub-degree refinement around the best coarse angles.
            fine_offsets = np.arange(-4.0, 4.001, 0.25)
            fine_angles: List[float] = []
            for phi in seed_angles:
                for dphi in fine_offsets.tolist():
                    cand = (phi + dphi) % 360.0
                    if any(_angular_distance_deg(cand, a) < 0.1 for a in covered | {360.0}):
                        continue
                    _append_unique_angle(fine_angles, cand, tol=0.05)

            for phi in fine_angles:
                result_id = _eval_identity(phi)
                if result_id is None:
                    continue
                if best is None or result_id.overall_confidence > best.overall_confidence:
                    best = result_id
                if result_id.overall_confidence >= self.confidence_threshold:
                    return best  # type: ignore[return-value]

            # If confidence is still weak, run a denser local sweep around all
            # FFT base hypotheses to avoid missing narrow high-confidence peaks.
            if best is None or best.overall_confidence < 0.75:
                dense_offsets = np.arange(-12.0, 12.001, 0.25)
                dense_angles: List[float] = []
                for phi in base_angles:
                    for dphi in dense_offsets.tolist():
                        cand = (phi + dphi) % 360.0
                        if any(_angular_distance_deg(cand, a) < 0.1 for a in covered | {360.0}):
                            continue
                        _append_unique_angle(dense_angles, cand, tol=0.01)

                for phi in dense_angles:
                    result_id = _eval_identity(phi)
                    if result_id is None:
                        continue
                    if best is None or result_id.overall_confidence > best.overall_confidence:
                        best = result_id
                    if result_id.overall_confidence >= self.confidence_threshold:
                        return best  # type: ignore[return-value]

            # Always include flips and 90° rotations for the top refined angles.
            identity_trials.sort(key=lambda t: t[0], reverse=True)
            top_trials = identity_trials[:2]
            inner_names: List[str] = ["hflip", "vflip", "rot270", "rot90", "rot180"]
            for _, phi in top_trials:
                de_rotated = rotate_cache.get(phi)
                if de_rotated is None:
                    de_rotated = _rotate_image(img, phi)
                    rotate_cache[phi] = de_rotated
                for name in inner_names:
                    try:
                        transformed = _apply_transform(de_rotated, name)
                        result = self._decode_once(
                            transformed,
                            transform=name,
                            rotation_angle=phi,
                        )
                    except ValueError:
                        continue

                    if best is None or result.overall_confidence > best.overall_confidence:
                        best = result
                    if result.overall_confidence >= self.confidence_threshold:
                        return best  # type: ignore[return-value]

        assert best is not None, "All candidate transforms failed (image too small?)."
        return best
