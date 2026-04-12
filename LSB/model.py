"""
model.py
========
Fully vectorized blind watermarking — confidence-weighted decoding,
optional CRC-32 error correction, optional adaptive NVF embedding.

Embedding rule
--------------
Each watermark bit shifts one randomly-selected pixel by ±alpha_eff::

    bit = 1  →  pixel += alpha_eff
    bit = 0  →  pixel -= alpha_eff

Without NVF, alpha_eff = alpha (constant).

With NVF, alpha_eff is spatially adaptive::

    alpha_eff(x) = (1 − NVF(x)) · alpha  +  NVF(x) · alpha_low

Confidence score (core decoding primitive)
------------------------------------------
For every embedded pixel position::

    score = center_pixel − mean(k×k neighbourhood, center excluded)

Positive → likely 1; negative → likely 0; |score| ≈ alpha_eff (for clean images).

When NVF is active, scores are normalised by the per-pixel alpha_eff before
averaging across copies, preventing high-strength pixels from dominating the
decision over low-strength ones::

    normalised_score_i = score_i / alpha_eff_i

Averaging normalised scores still improves SNR proportionally to √(copies).

Noise Visibility Function (NVF)
--------------------------------
NVF exploits the human visual system's reduced sensitivity to perturbations in
textured or noisy image regions.  Local variance ϑ²(x) is computed via a box
filter in the Fourier domain (fully vectorized across channels)::

    ϑ²(x)   = E[pixel²](x) − E[pixel]²(x)          (local, W×W window)
    NVF(x)  = 1 / (1 + D · ϑ²(x) / max ϑ²)

    NVF ≈ 0  →  high variance / textured region  →  alpha_eff ≈ alpha  (strong)
    NVF ≈ 1  →  low variance  / smooth region    →  alpha_eff ≈ alpha_low (soft)

The per-pixel alpha map lets the encoder push more energy into areas where
distortions are perceptually invisible while staying imperceptible in flat areas.

Payload layouts
---------------
CRC disabled  (``use_crc=False``)::

    payload = tile(message, msg_repeat)

    Decode: reshape scores to (msg_repeat, message_length), average, hard-decide.

CRC enabled   (``use_crc=True``)::

    payload = [ tile(message, msg_repeat)  |  tile(CRC-32, crc_repeat) ]
                    m copies                       n copies,  n > m

    Decode pipeline:
      1. Normalise scores by alpha_eff (NVF mode) or leave as-is.
      2. Average message scores over m copies → mean score per bit.
      3. Average CRC scores over n copies → hard-decode reference CRC.
      4. Hard-decide message from sign of mean scores.
      5. Verify CRC-32(message) == reference CRC → return on success.
      6. Brute-force repair: flip the k = 1 … *max_flip_bits* least-confident
         message bits (sorted by |mean score| ascending) until CRC passes.

Why CRC-32 even for a 32-bit message?
--------------------------------------
* **Silent errors.** Without a verifiable reference you cannot know whether
  the decoded output is correct.  CRC makes errors detectable.

* **Brute-force fidelity.** With k=3 and a pool of 32 bits, the search has
  ≤ 4 960 candidates; CRC-32 gives a false-acceptance probability of
  4 960 × 2⁻³² ≈ 10⁻⁶.  CRC-16 would push that to ~5 % — unacceptable.

Fourier-domain filtering
------------------------
Local mean and mean-of-squares are both computed via FFT convolution with a
normalised box kernel of size W×W.  The kernel is embedded in a zero-padded
(H, W) array with the origin rolled to (0, 0) for correct circular
convolution.  All image channels are processed simultaneously using NumPy's
batched ``rfft2`` / ``irfft2``.

Dependencies
------------
numpy, opencv-python (cv2 available for callers; not used internally).
"""

from __future__ import annotations

import binascii
import itertools
from typing import Optional, Tuple

import cv2          # noqa: F401 — available for callers (e.g. PSNR checks)
import numpy as np


# ── Module-level CRC helper ───────────────────────────────────────────────────

_CRC_BITS: int = 32   # fixed width of the CRC field


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
    Vectorized blind watermarking — confidence-weighted, optionally CRC-protected,
    optionally NVF-adaptive.

    Parameters
    ----------
    message_length : int, default 32
        Number of bits in the watermark message.
    key : int, default 42
        PRNG seed for deterministic pixel selection.
        **Must be identical at encode and decode time.**
    alpha : float, default 95.0
        Base embedding strength (applied to high-variance / textured regions
        when NVF is enabled; applied uniformly otherwise).
        Larger → more robust watermark, lower PSNR.
    use_crc : bool, default False
        Enable CRC-32 error detection and brute-force correction.
        When *False* the entire embedding budget goes to message copies;
        compensate with a higher *msg_repeat* (e.g. 8–10).
    msg_repeat : int, default 35
        Number of message copies embedded (m).
        Also the sole redundancy knob when ``use_crc=False``.
        When ``use_crc=True``, must satisfy ``msg_repeat < crc_repeat``.
    crc_repeat : int, default 5
        Number of CRC-32 copies embedded (n).  Ignored when ``use_crc=False``.
        Must satisfy ``crc_repeat > msg_repeat``.
    neighborhood_size : int, default 5
        Side length of the square neighbourhood for the confidence score.
        Must be an odd integer ≥ 3 (e.g. 3 for 3×3, 5 for 5×5).
    max_flip_bits : int, default 3
        Maximum bits flipped per brute-force repair level (CRC mode only).
        k=3 → ≤ 4 960 CRC checks (negligible runtime).
    use_nvf : bool, default True
        Enable adaptive NVF-based embedding strength.
        When *True*, each embedded pixel is modulated by its local NVF value
        so that flat regions receive weaker perturbations (less visible) while
        textured regions receive stronger ones (still invisible).
    nvf_window_size : int, default 5
        Side length W of the square window used for local variance estimation.
        Larger windows capture broader texture context.
        Must be a positive odd integer.
    nvf_D : float, default 75.0
        Sensitivity parameter D in NVF = 1 / (1 + D · ϑ²/max ϑ²).
        Larger D → steeper contrast between smooth and textured regions;
        smoother areas get a proportionally smaller alpha.
    nvf_alpha_low : float or None, default 20.0
        Embedding strength applied to the smoothest regions (NVF ≈ 1).
        When *None* this defaults to ``alpha / 3``, matching the 3× ratio
        recommended in the perceptual watermarking literature.
        Must be < *alpha* (otherwise NVF has no effect).

    Total embedded bits
    -------------------
    ``use_crc=False`` :  ``message_length × msg_repeat``
    ``use_crc=True``  :  ``message_length × msg_repeat + 32 × crc_repeat``

    Notes
    -----
    The NVF map is computed once during ``encode`` (on the original image) and
    once during ``decode`` (on the watermarked image, as a blind approximation).
    Because the perturbation magnitudes are small relative to local texture
    energy, the approximation is tight in practice.
    """

    def __init__(
        self,
        message_length: int       = 32,
        key: int                  = 42,
        alpha: float              = 95.0,
        use_crc: bool             = False,
        msg_repeat: int           = 35,
        crc_repeat: int           = 5,
        neighborhood_size: int    = 5,
        max_flip_bits: int        = 3,
        *,                               # NVF params are keyword-only
        use_nvf: bool             = True,
        nvf_window_size: int      = 5,
        nvf_D: float              = 75.0,
        nvf_alpha_low: Optional[float] = 20.0,
    ) -> None:
        # ── Validation ───────────────────────────────────────────────────
        if neighborhood_size % 2 == 0 or neighborhood_size < 3:
            raise ValueError("neighborhood_size must be an odd integer ≥ 3.")
        if use_crc and crc_repeat <= msg_repeat:
            raise ValueError(
                f"crc_repeat ({crc_repeat}) must exceed msg_repeat ({msg_repeat}) "
                "so the CRC field is decoded more reliably than the message."
            )
        if nvf_window_size < 1 or nvf_window_size % 2 == 0:
            raise ValueError("nvf_window_size must be a positive odd integer.")
        if nvf_D <= 0:
            raise ValueError("nvf_D must be strictly positive.")

        resolved_alpha_low = float(nvf_alpha_low) if nvf_alpha_low is not None \
                             else float(alpha) / 3.0
        if use_nvf and resolved_alpha_low >= float(alpha):
            raise ValueError(
                f"nvf_alpha_low ({resolved_alpha_low:.3g}) must be strictly less "
                f"than alpha ({alpha:.3g}); otherwise NVF has no effect."
            )

        # ── Store attributes ─────────────────────────────────────────────
        self.message_length    = message_length
        self.key               = key
        self.alpha             = float(alpha)
        self.use_crc           = use_crc
        self.msg_repeat        = msg_repeat
        self.crc_repeat        = crc_repeat
        self.neighborhood_size = neighborhood_size
        self.max_flip_bits     = max_flip_bits
        self.use_nvf           = use_nvf
        self.nvf_window_size   = nvf_window_size
        self.nvf_D             = float(nvf_D)
        self.nvf_alpha_low     = resolved_alpha_low

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
        ValueError
            If the image is too small to hold the payload.
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

        Parameters
        ----------
        img : np.ndarray, shape (H, W, C), float32
        rows, cols, chans : np.ndarray[intp], shape (_total_embedded,)

        Returns
        -------
        np.ndarray, shape (_total_embedded,), dtype float32
            Positive → bit likely 1; negative → likely 0;
            larger |score| → higher confidence (≈ alpha_eff for clean images).
        """
        padded = np.pad(
            img,
            ((self._pad, self._pad), (self._pad, self._pad), (0, 0)),
            mode="reflect",
        )
        # Neighbour absolute indices into padded array: (n_bits, n_neighbors)
        nr  = rows[:, np.newaxis]  + self._pad + self._DR   # (N, k²−1)
        nc  = cols[:, np.newaxis]  + self._pad + self._DC   # (N, k²−1)
        nch = np.broadcast_to(chans[:, np.newaxis], nr.shape).astype(np.intp)

        return img[rows, cols, chans] - padded[nr, nc, nch].mean(axis=1)

    # ── NVF ──────────────────────────────────────────────────────────────

    def _fft_box_filter(
        self,
        img_chw: np.ndarray,
        K_fft: np.ndarray,
        H: int,
        W: int,
    ) -> np.ndarray:
        """
        Apply a box filter to all channels simultaneously via FFT convolution.

        Parameters
        ----------
        img_chw : np.ndarray, shape (C, H, W), float32
            Input image with channels-first layout.
        K_fft : np.ndarray, shape (H, W//2+1), complex64
            Pre-computed rfft2 of the normalised box kernel (origin at (0,0)).
        H, W : int
            Spatial dimensions (needed for irfft2 output shape).

        Returns
        -------
        np.ndarray, shape (C, H, W), float32
            Convolution result (locally averaged values).

        Notes
        -----
        Implements *circular* convolution; wrap-around artefacts at image
        borders are limited to a strip of width ``nvf_window_size // 2``
        pixels, which is negligible for typical window sizes.
        """
        IMG_fft = np.fft.rfft2(img_chw)                       # (C, H, W//2+1)
        filtered = np.fft.irfft2(IMG_fft * K_fft, s=(H, W))  # (C, H, W)
        return filtered.astype(np.float32)

    def _compute_nvf(self, img: np.ndarray) -> np.ndarray:
        """
        Compute the Noise Visibility Function map via FFT-based box filtering.

        Local variance ϑ²(x) at each pixel is estimated as::

            ϑ²(x) = E_W[pixel²](x) − E_W[pixel]²(x)

        where E_W denotes expectation over a W×W sliding window.
        Both expectations are computed through Fourier-domain convolution
        with a normalised box kernel, making the operation fully vectorized
        across spatial dimensions *and* channels in a single pass.

        The NVF is then::

            NVF(x) = 1 / (1 + D · ϑ²(x) / max_x ϑ²(x))

        NVF ∈ (0, 1]:
            → 1 in perfectly flat regions (zero variance)
            → 0 as local variance → ∞ relative to the image maximum

        Parameters
        ----------
        img : np.ndarray, shape (H, W, C), float32

        Returns
        -------
        np.ndarray, shape (H, W, C), float32
            NVF map, values in (0, 1].  Returns an all-ones array if the
            image has zero variance everywhere (e.g. a solid-colour image).
        """
        H, W, C = img.shape
        ws       = self.nvf_window_size

        # ── Build box-filter kernel, origin at (0,0) for FFT convolution ──
        # Place the ws×ws uniform kernel in the top-left corner, then roll
        # so its effective centre is at position (0, 0).  This ensures that
        # the circular convolution aligns correctly.
        kernel            = np.zeros((H, W), dtype=np.float32)
        kernel[:ws, :ws]  = 1.0 / (ws * ws)
        half              = ws // 2
        kernel            = np.roll(kernel, (-half, -half), axis=(0, 1))
        K_fft             = np.fft.rfft2(kernel)               # (H, W//2+1)

        # ── Channels-first layout for batched FFT ─────────────────────────
        img_chw = img.transpose(2, 0, 1).astype(np.float32)    # (C, H, W)

        mean    = self._fft_box_filter(img_chw,        K_fft, H, W)  # E[x]
        mean_sq = self._fft_box_filter(img_chw ** 2,   K_fft, H, W)  # E[x²]

        # Local variance: clip negatives from floating-point cancellation
        var = np.maximum(mean_sq - mean ** 2, 0.0)              # (C, H, W)

        # Per-channel maximum variance for normalisation
        max_var = var.max(axis=(1, 2), keepdims=True)           # (C, 1, 1)

        # Threshold below which a channel is considered perceptually flat.
        # FFT arithmetic on float32 introduces residual noise ≈ 1e-3 intensity²
        # even on constant images; 1.0 intensity² is well above that floor yet
        # negligibly small relative to any real texture (variance routinely
        # reaches hundreds to thousands for natural images in [0, 255]).
        _FLAT_THRESHOLD = 1.0

        is_flat  = max_var < _FLAT_THRESHOLD                    # (C, 1, 1) bool
        safe_max = np.where(is_flat, 1.0, max_var)

        nvf_chw  = 1.0 / (1.0 + self.nvf_D * var / safe_max)   # (C, H, W)

        # Flat channels (zero / near-zero variance) → NVF = 1 everywhere,
        # which maps to alpha_low (the most conservative choice).
        nvf_chw  = np.where(is_flat, 1.0, nvf_chw)

        return nvf_chw.transpose(1, 2, 0)                        # (H, W, C)

    def _alpha_map(
        self,
        img: np.ndarray,
        rows: np.ndarray,
        cols: np.ndarray,
        chans: np.ndarray,
    ) -> np.ndarray:
        """
        Compute per-position effective embedding strength when NVF is active.

        Samples the NVF at every embedded pixel position and maps it to an
        alpha value::

            alpha_eff_i = (1 − NVF_i) · alpha  +  NVF_i · alpha_low

        This is a convex combination: NVF = 0 (textured) → alpha (strong);
        NVF = 1 (smooth) → alpha_low (soft).

        Parameters
        ----------
        img : np.ndarray, shape (H, W, C), float32
        rows, cols, chans : np.ndarray[intp], shape (_total_embedded,)

        Returns
        -------
        np.ndarray, shape (_total_embedded,), float32
            Per-position embedding strength.
        """
        nvf     = self._compute_nvf(img)            # (H, W, C)
        nvf_pts = nvf[rows, cols, chans]            # (_total_embedded,)
        return ((1.0 - nvf_pts) * self.alpha
                + nvf_pts       * self.nvf_alpha_low).astype(np.float32)

    # ── CRC repair ───────────────────────────────────────────────────────

    def _crc_repair(
        self,
        msg_bits: np.ndarray,
        confidence: np.ndarray,
        ref_crc: np.ndarray,
    ) -> np.ndarray:
        """
        Flip the fewest, least-confident bits until CRC-32 passes.

        Levels k = 1, 2, …, *max_flip_bits* are tried in ascending order.
        At level k, all C(pool, k) combinations of k indices from the
        ``pool`` least-confident positions are exhaustively tested.
        The pool is capped at 32 so the worst-case cost (k = 3) is
        C(32, 3) = 4 960 CRC checks, regardless of ``message_length``.

        Parameters
        ----------
        msg_bits : np.ndarray, shape (message_length,), uint8
            Current best-effort hard decision.
        confidence : np.ndarray, shape (message_length,), float32
            Per-bit confidence magnitudes (lower → less certain → flip first).
        ref_crc : np.ndarray, shape (32,), uint8
            Reference CRC-32 decoded from the embedded CRC copies.

        Returns
        -------
        np.ndarray, shape (message_length,), uint8
            Corrected bits, or *msg_bits* unchanged if repair is exhausted.
        """
        pool = np.argsort(confidence)[:min(self.message_length, 32)].tolist()

        for k in range(1, self.max_flip_bits + 1):
            for idx in itertools.combinations(pool, k):
                trial          = msg_bits.copy()
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

        Payload layout::

            use_crc=False : tile(message, msg_repeat)
            use_crc=True  : [ tile(message, msg_repeat) | tile(CRC-32, crc_repeat) ]

        Embedding perturbation for position i::

            Δ_i = alpha_eff_i × (2 × bit_i − 1)       # +alpha_eff for 1, −alpha_eff for 0

        When ``use_nvf=False``, alpha_eff_i = alpha (constant).
        When ``use_nvf=True``,  alpha_eff_i ∈ [alpha_low, alpha] depending on
        local image texture (NVF-weighted; see class docstring).

        Output is clipped to [0, 255].

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C), uint8 or float
            Cover image.  Grayscale (2-D) and colour (3-D) are both supported.
        watermark : np.ndarray, shape (message_length,), dtype uint8
            Binary watermark message (values must be 0 or 1).

        Returns
        -------
        np.ndarray
            Watermarked image, **same shape and dtype** as *image*.
        """
        orig_dtype = image.dtype
        grayscale  = image.ndim == 2
        img        = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        # ── Build payload ─────────────────────────────────────────────────
        wm      = np.asarray(watermark, dtype=np.uint8)
        payload = (
            np.concatenate([
                np.tile(wm, self.msg_repeat),
                np.tile(_crc32_bits(wm), self.crc_repeat),
            ])
            if self.use_crc else
            np.tile(wm, self.msg_repeat)
        )  # (_total_embedded,)

        rows, cols, chans = self._pixel_indices(H, W, C)

        # ── Per-position embedding strength ───────────────────────────────
        if self.use_nvf:
            alpha_eff = self._alpha_map(img, rows, cols, chans)  # (_total_embedded,)
        else:
            alpha_eff = self.alpha                                # scalar broadcast

        # ── Embed: Δ = alpha_eff × (2·bit − 1) ───────────────────────────
        delta = alpha_eff * (2.0 * payload.astype(np.float32) - 1.0)
        img[rows, cols, chans] += delta
        np.clip(img, 0.0, 255.0, out=img)

        if grayscale:
            img = img[:, :, 0]
        return img.astype(orig_dtype)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the binary watermark from *image* (blind decoding).

        Only the ``_total_embedded`` selected pixels and their k×k neighbours
        are accessed — no full-image scan is required.

        **NVF disabled** — standard confidence-score decoding:

            Average raw confidence scores over copies → hard-decide from sign.

        **NVF enabled** — alpha_eff-normalised decoding:

            The NVF is recomputed on the watermarked image (blind approximation;
            tight because perturbations are small relative to local texture).
            Raw confidence scores are then normalised by the per-position
            alpha_eff before averaging, removing the influence of spatially
            varying embedding strength on the soft decision::

                normalised_score_i = score_i / alpha_eff_i

        **CRC enabled** (either NVF mode):

            After averaging, verify CRC-32; on failure run brute-force repair
            on the least-confident message bits (see ``_crc_repair``).

        Parameters
        ----------
        image : np.ndarray, shape (H, W) or (H, W, C)
            Watermarked image.  Dtype can be uint8 or float.

        Returns
        -------
        np.ndarray, shape (message_length,), dtype uint8
            Decoded binary watermark.
        """
        grayscale = image.ndim == 2
        img       = image.astype(np.float32)
        if grayscale:
            img = img[:, :, np.newaxis]
        H, W, C = img.shape

        rows, cols, chans = self._pixel_indices(H, W, C)
        scores = self._confidence_scores(img, rows, cols, chans)  # (_total_embedded,)

        # ── Optional NVF normalisation ────────────────────────────────────
        # Divide each raw score by its effective alpha so that the subsequent
        # averaging treats every copy equally, regardless of local texture.
        if self.use_nvf:
            alpha_eff = self._alpha_map(img, rows, cols, chans)   # (_total_embedded,)
            # Guard against near-zero alpha (extremely unlikely but safe)
            scores = scores / np.maximum(alpha_eff, 1e-6)

        # ── Average message scores over msg_repeat copies ─────────────────
        msg_mean = (
            scores[: self.message_length * self.msg_repeat]
            .reshape(self.msg_repeat, self.message_length)
            .mean(axis=0)                                   # (message_length,)
        )
        msg_bits = (msg_mean >= 0.0).astype(np.uint8)       # hard decision
        msg_conf = np.abs(msg_mean)                          # per-bit confidence

        if not self.use_crc:
            return msg_bits

        # ── Average CRC scores over crc_repeat copies ─────────────────────
        ref_crc = (
            scores[self.message_length * self.msg_repeat :]
            .reshape(self.crc_repeat, _CRC_BITS)
            .mean(axis=0)
            >= 0.0
        ).astype(np.uint8)                                   # (32,)

        # ── Verify → repair → return ──────────────────────────────────────
        if np.array_equal(_crc32_bits(msg_bits), ref_crc):
            return msg_bits

        return self._crc_repair(msg_bits, msg_conf, ref_crc)
