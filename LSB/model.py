"""
model.py
========
Vectorized blind watermarking with 8-way (D4) spatial symmetry and
optional brute-force rotation search.

Embedding rule
--------------
Each watermark bit shifts one randomly-selected pixel by ±alpha::

    bit = 1  →  pixel += alpha
    bit = 0  →  pixel -= alpha

The same bit is embedded at **8 symmetric pixel positions** derived from
a single seed, so the watermark pattern is invariant under the full D4
group of square symmetries (identity, three 90°-step rotations, and four
reflections).  Robustness to flips and 90°/180°/270° rotations therefore
requires **no explicit search** — it is guaranteed by construction.

8-way (D4) symmetry
-------------------
For each seed position (r, c) within the centred S×S square (S = min(H,W),
s = S−1), the eight D4 images are::

    (r,   c  )   identity
    (c,   s−r)   rotation  90° CW
    (s−r, s−c)   rotation 180°
    (s−c, r  )   rotation 270° CW
    (r,   s−c)   horizontal flip
    (s−r, c  )   vertical flip
    (c,   r  )   transpose        (reflect across main diagonal)
    (s−c, s−r)   anti-transpose   (reflect across anti-diagonal)

All eight positions carry the identical payload bit.  After any D4 transform,
the set of watermarked pixels is merely permuted — the majority-vote decoder
reads the same bits as on the untransformed image.

Decoding — two quality metrics
--------------------------------
**Metric 1 — Embedding strength** (``embedding_strength``)::

    strength = mean_i( |avg_score_i| )

where avg_score_i is the mean of the eight D4-copy scores for seed i.
Near zero → wrong pixel positions (wrong angle) or nothing embedded.

**Metric 2 — Overall confidence** (``overall_confidence``)::

    For each bit position k ∈ [0, message_length):
        frac_k  = fraction of msg_repeat copies that decode bit k as 1
        conf_k  = max(frac_k, 1 − frac_k)       ∈ [0.5, 1.0]

    overall_confidence = mean_k( conf_k )        ∈ [0.5, 1.0]

Measures copy agreement per bit, then averages.  1.0 = every copy agrees
on every bit (perfect decode).  0.5 = purely random (wrong orientation or
no watermark present).

Angle search
------------
When ``angle_step`` is set, the decoder tries rotations in the half-open
interval [0°, 90°) at ``angle_step`` increments::

    0°, angle_step, 2·angle_step, …

D4 symmetry already handles multiples of 90°, so searching [0°, 90°) is
sufficient to cover all 360°: for any attack angle θ_attack, the value
θ_search = (−θ_attack) mod 90° lies in [0°, 90°), and after applying
θ_search the residual offset is a multiple of 90° which D4 corrects.

Early exit fires as soon as ``overall_confidence ≥ confidence_threshold``.

Dependencies
------------
numpy, opencv-python (cv2 re-exported for callers).
"""

from __future__ import annotations

from typing import NamedTuple, Optional, Tuple

import cv2          # noqa: F401 — re-exported for callers (e.g. PSNR checks)
import numpy as np


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
        Mean |averaged score| at each seed position (Metric 1).
        Scales with alpha; near zero → wrong angle or nothing embedded.
    overall_confidence : float
        Mean per-bit copy-agreement fraction (Metric 2), in [0.5, 1.0].
        1.0 → every copy agrees on every bit (perfect decode).
        0.5 → fully random (wrong angle or no watermark).
    angle : float
        Rotation angle (degrees) applied before decoding.
        0.0 when no angle search is performed.
    """
    bits:               np.ndarray
    embedding_strength: float
    overall_confidence: float
    angle:              float


# ─────────────────────────────────────────────────────────────────────────────
# Rotation helper
# ─────────────────────────────────────────────────────────────────────────────

def _rotate_image(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    Rotate a float32 image CCW by ``angle_deg`` around its centre.

    Uses bilinear interpolation and reflect padding to avoid border
    artefacts near the edges.

    Parameters
    ----------
    img       : np.ndarray, shape (H, W, C), float32
    angle_deg : float — counter-clockwise rotation in degrees

    Returns
    -------
    np.ndarray, shape (H, W, C), float32
    """
    H, W = img.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), angle_deg, 1.0)
    return cv2.warpAffine(
        img, M, (W, H),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )


# ─────────────────────────────────────────────────────────────────────────────
# WatermarkModel
# ─────────────────────────────────────────────────────────────────────────────

class WatermarkModel:
    """
    Vectorized blind watermarking with 8-way (D4) symmetry.

    The watermark is embedded with 8-fold spatial symmetry — identical bits
    appear at all D4-symmetric pixel positions — making it inherently robust
    to horizontal/vertical flips and 90°/180°/270° rotations without any
    explicit transform search.

    For arbitrary-angle robustness, an optional brute-force search tries
    rotations in [0°, 90°) at a configurable step size, with early exit
    when the confidence threshold is reached.

    Parameters
    ----------
    message_length : int, default 32
        Number of bits in the watermark message.
    key : int, default 42
        PRNG seed for deterministic pixel selection.
        **Must be identical at encode and decode time.**
    alpha : float, default 80.0
        Embedding strength.  Larger → more robust, lower PSNR.
    msg_repeat : int, default 5
        Number of independent message copies embedded (for majority vote).
    neighborhood_size : int, default 5
        Side length of the square neighbourhood ring used for confidence
        scoring.  Must be an odd integer ≥ 3.
    angle_step : float or None, default 1.0
        Step size in degrees for the brute-force angle search over [0°, 90°).
        Set to ``None`` to skip the search entirely (D4 robustness is still
        guaranteed by construction).
    confidence_threshold : float, default 0.80
        Early-exit threshold for the angle search.  A result with
        ``overall_confidence ≥ confidence_threshold`` is accepted
        immediately (range (0.5, 1.0]).

    Notes
    -----
    Total physical pixels modified per encode:
        ``message_length × msg_repeat × 8``

    Each of the 8 copies per seed is embedded at a D4-symmetric position,
    so the effective independent vote count at decode time is ``msg_repeat``.
    """

    def __init__(
        self,
        message_length:       int            = 32,
        key:                  int            = 42,
        alpha:                float          = 80.0,
        msg_repeat:           int            = 5,
        neighborhood_size:    int            = 5,
        angle_step:           Optional[float] = 1.0,
        confidence_threshold: float          = 0.80,
    ) -> None:
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")

        self.message_length       = message_length
        self.key                  = key
        self.alpha                = float(alpha)
        self.msg_repeat           = msg_repeat
        self.neighborhood_size    = neighborhood_size
        self.angle_step           = angle_step
        self.confidence_threshold = confidence_threshold

        # Number of unique (seed, message-bit) decisions.
        self._n_seeds: int = message_length * msg_repeat
        # Each seed maps to 8 physical pixels (D4 symmetry).
        self._n_positions: int = self._n_seeds * 8

        # ── Neighbourhood ring offsets — precomputed once ─────────────────
        # All (Δrow, Δcol) pairs in the k×k window except the centre (0, 0).
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
        Generate all D4-symmetric pixel positions for the (H, W, C) image.

        A square of side ``S = min(H, W)`` is centred in the image; all
        watermarked pixels are confined to this square so that 90°-rotation
        symmetry maps the square onto itself.

        ``_n_seeds`` seed positions (r, c) are drawn uniformly from the S×S
        grid without replacement; their 8 D4 images are then computed::

            (r,   c  )   identity
            (c,   s−r)   rotation  90° CW
            (s−r, s−c)   rotation 180°
            (s−c, r  )   rotation 270° CW
            (r,   s−c)   horizontal flip
            (s−r, c  )   vertical flip
            (c,   r  )   transpose        (reflect across main diagonal)
            (s−c, s−r)   anti-transpose   (reflect across anti-diagonal)

        The output arrays are laid out so that indices
        ``[8k .. 8k+7]`` correspond to the 8 D4 images of seed k, all
        sharing the same channel ``ch[k]`` and the same payload bit.

        Parameters
        ----------
        H, W, C : int — image dimensions

        Returns
        -------
        rows  : np.ndarray, shape (_n_positions,), intp
        cols  : np.ndarray, shape (_n_positions,), intp
        chans : np.ndarray, shape (_n_positions,), intp

        Raises
        ------
        ValueError
            If the image is too small to accommodate all seed positions.
        """
        S     = min(H, W)              # square side for D4 symmetry
        r_off = (H - S) // 2           # row offset to centre the square
        c_off = (W - S) // 2           # col offset
        s     = S - 1                  # maximum index within the square

        if self._n_seeds > S * S:
            raise ValueError(
                f"Image too small: need {self._n_seeds} seed positions but "
                f"the centred {S}×{S} square holds only {S * S}."
            )

        rng  = np.random.default_rng(self.key)
        flat = rng.choice(S * S, size=self._n_seeds, replace=False)
        sr, sc = np.unravel_index(flat, (S, S))   # (n_seeds,) in [0, S)
        ch     = rng.integers(0, C, size=self._n_seeds)

        # 8 D4 transforms of (sr, sc) in S×S coordinates.
        # all_r / all_c: shape (8, n_seeds).
        # Transposed to (n_seeds, 8) and ravelled → positions [8k..8k+7]
        # are the 8 transforms of seed k (groups of 8 consecutive entries).
        all_r = np.stack([sr,   sc,   s - sr, s - sc,
                           sr,   s - sr, sc,   s - sc])   # (8, n_seeds)
        all_c = np.stack([sc,   s - sr, s - sc, sr,
                           s - sc, sc,   sr,   s - sr])   # (8, n_seeds)

        rows  = (all_r + r_off).T.ravel().astype(np.intp)   # (n_positions,)
        cols  = (all_c + c_off).T.ravel().astype(np.intp)   # (n_positions,)
        chans = np.repeat(ch, 8).astype(np.intp)             # (n_positions,)
        return rows, cols, chans

    def _confidence_scores(
        self,
        img:   np.ndarray,
        rows:  np.ndarray,
        cols:  np.ndarray,
        chans: np.ndarray,
    ) -> np.ndarray:
        """
        Compute a signed score at each pixel position::

            score_i = img[row_i, col_i, chan_i]
                    − mean( k×k neighbourhood, centre excluded )

        Positive score → decoded bit = 1; negative → bit = 0.
        Magnitude encodes how far the pixel deviates from its local mean,
        i.e. how confidently it was shifted during embedding.

        Reflect-padding (``_pad`` pixels wide) handles border pixels without
        introducing artificial edges.

        Parameters
        ----------
        img   : np.ndarray, shape (H, W, C), float32
        rows  : np.ndarray, shape (N,), intp
        cols  : np.ndarray, shape (N,), intp
        chans : np.ndarray, shape (N,), intp

        Returns
        -------
        np.ndarray, shape (N,), float32
        """
        padded = np.pad(
            img,
            ((self._pad, self._pad), (self._pad, self._pad), (0, 0)),
            mode="reflect",
        )
        nr  = rows[:, None] + self._pad + self._DR          # (N, k²−1)
        nc  = cols[:, None] + self._pad + self._DC
        nch = np.broadcast_to(chans[:, None], nr.shape).astype(np.intp)

        return img[rows, cols, chans] - padded[nr, nc, nch].mean(axis=1)

    def _decode_once(
        self, img: np.ndarray, angle: float = 0.0
    ) -> DecodeResult:
        """
        Decode the watermark from a single (H, W, C) float32 image.

        The ``_n_positions = 8 × _n_seeds`` raw scores are reshaped to
        ``(_n_seeds, 8)`` and averaged along axis 1, collapsing the 8 D4
        copies of each seed into a single consensus score.  The resulting
        ``(_n_seeds,)`` vector is reshaped to ``(msg_repeat, message_length)``
        for the majority-vote step (one row per copy, one column per bit).

        Parameters
        ----------
        img   : np.ndarray, shape (H, W, C), float32
        angle : float — echoed into the returned DecodeResult

        Returns
        -------
        DecodeResult
        """
        H, W, C = img.shape
        rows, cols, chans = self._pixel_indices(H, W, C)

        # Raw scores at all 8 × n_seeds positions.
        raw_scores = self._confidence_scores(img, rows, cols, chans)

        # Average the 8 D4 copies of each seed → consensus score per seed.
        # Shape: (_n_positions,) → (_n_seeds, 8) → mean → (_n_seeds,)
        scores = raw_scores.reshape(self._n_seeds, 8).mean(axis=1)

        # ── Metric 1: embedding strength ──────────────────────────────────
        embedding_strength = float(np.abs(scores).mean())

        # ── Metric 2: per-bit copy agreement → overall confidence ─────────
        scores_2d  = scores.reshape(self.msg_repeat, self.message_length)
        bits_2d    = scores_2d >= 0.0                        # (repeat, len)
        frac_one   = bits_2d.mean(axis=0)                    # (len,)
        bit_conf   = np.maximum(frac_one, 1.0 - frac_one)   # (len,)
        overall_confidence = float(bit_conf.mean())

        # Final bits: majority vote.
        bits = (frac_one >= 0.5).astype(np.uint8)

        return DecodeResult(
            bits=bits,
            embedding_strength=embedding_strength,
            overall_confidence=overall_confidence,
            angle=angle,
        )

    # ── Public API ────────────────────────────────────────────────────────

    def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray:
        """
        Embed *watermark* into *image* with 8-fold D4 spatial symmetry.

        Each selected seed position and all seven of its D4 images receive
        the same additive shift::

            Δ = alpha × (2 × bit − 1)   # +alpha for bit=1, −alpha for bit=0

        The full payload is ``tile(watermark, msg_repeat)`` (one bit per
        seed), repeated 8 times per seed to fill all D4 positions.
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

        # One bit per seed: tile the message across all msg_repeat copies.
        payload     = np.tile(np.asarray(watermark, dtype=np.uint8), self.msg_repeat)
        # Expand: each seed bit is repeated 8 times for its D4 positions.
        # Shape: (_n_seeds,) → (_n_positions,).  Positions [8k..8k+7] share bit k.
        payload_sym = np.repeat(payload, 8).astype(np.float32)

        rows, cols, chans         = self._pixel_indices(H, W, C)
        img[rows, cols, chans]   += self.alpha * (2.0 * payload_sym - 1.0)
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the watermark from *image*; return bits only.

        Thin wrapper around :meth:`decode_verbose` — use that method for
        quality metrics and the best-matching angle.

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
        Extract the watermark with full diagnostic information.

        D4 robustness
            Flips and 90°/180°/270° rotations are handled by construction —
            no explicit search is needed for these transforms.

        Angle search (when ``angle_step`` is not None)
            For each candidate angle θ ∈ {0°, angle_step, 2·angle_step, …}
            with θ < 90°:

            1. Rotate the image CCW by θ using bilinear interpolation.
            2. Decode and compute ``overall_confidence``.
            3. **Early exit** if ``overall_confidence ≥ confidence_threshold``.
            4. After all candidates, return the highest-confidence result.

            Searching [0°, 90°) is sufficient: for any attack rotation θ_attack,
            the candidate θ_search = (−θ_attack) mod 90° exists in this range,
            and the remaining 90°-multiple offset is corrected by D4 symmetry.

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)

        Returns
        -------
        DecodeResult — bits, embedding_strength, overall_confidence, angle
        """
        grayscale = image.ndim == 2
        img       = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]

        # ── Fast path: no angle search ─────────────────────────────────────
        if self.angle_step is None:
            return self._decode_once(img, angle=0.0)

        # ── Angle search over [0°, 90°) ────────────────────────────────────
        result = self._decode_once(img, angle=0.0)
        if result.overall_confidence >= self.confidence_threshold:
            return result  # early exit on unrotated image
        angles = np.arange(0.0, 90.0, float(self.angle_step))
        best: Optional[DecodeResult] = None

        for angle in angles:
            # Avoid a redundant copy at angle = 0°.
            rotated = img if angle == 0.0 else _rotate_image(img, angle)

            try:
                result = self._decode_once(rotated, angle=float(angle))
            except ValueError:
                continue  # image too small after rotation (rare edge case)

            if best is None or result.overall_confidence > best.overall_confidence:
                best = result

            if result.overall_confidence >= self.confidence_threshold:
                break  # reliable decode found — skip remaining candidates

        assert best is not None, "Angle search produced no valid result."
        return best
