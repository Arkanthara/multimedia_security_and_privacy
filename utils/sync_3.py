"""
utils/sync.py
=============
Frequency-domain synchronisation for LSB watermark decoding.

Pipeline
--------
1. Fourier-Mellin transform (FFT magnitude → log-polar) on image and reference
2. Phase correlation in log-polar space  → estimate rotation + scale
3. Correct rotation / scale             → cropped (no resize artefacts)
4. Cartesian phase correlation          → estimate translation
5. Correct translation                  → final aligned image (NaN-padded)
"""

import numpy as np
from skimage.filters import window
from skimage.transform import rescale, rotate, warp, SimilarityTransform, warp_polar
from skimage.registration import phase_cross_correlation
from utils.patch import build_reference_patch


# ---------------------------------------------------------------------------
# Fourier-Mellin magnitude spectrum
# ---------------------------------------------------------------------------

def _fourier_mellin_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the Fourier-Mellin magnitude spectrum of *image*.

    Steps
    -----
    1. Apply a Hanning window to suppress edge artefacts.
    2. Compute the 2-D FFT and shift DC to the centre.
    3. Zero the DC component to suppress low-frequency dominance.
    4. Take the log-magnitude to compress the dynamic range.
    5. Normalize to [0, 1] for stable phase correlation.
    6. Resample to log-polar coordinates (bicubic interpolation).

    The resulting representation is invariant to translation (magnitude
    discards phase) and transforms rotation/scale into shifts, making it
    directly suitable for phase correlation.

    Parameters
    ----------
    image : ndarray of shape (H, W), real-valued

    Returns
    -------
    log_polar : ndarray of shape (H, W), float64
        Log-polar Fourier-Mellin spectrum, ready for phase correlation.
    """
    H, W = image.shape

    # 1. Hanning window — reduces spectral leakage from non-periodic borders
    win = window("hann", image.shape)
    windowed = image * win

    # 2. FFT magnitude, DC centred
    f = np.fft.fftshift(np.fft.fft2(windowed))
    magnitude = np.abs(f)

    # 3. Zero DC component (centre pixel) — avoids its large value skewing
    #    the phase correlation peak
    magnitude[H // 2, W // 2] = 0.0

    # 4. Log-magnitude — compresses dynamic range so high-frequency content
    #    is not drowned out by the bright low-frequency ring
    log_mag = np.log1p(magnitude)

    # 5. Normalize to [0, 1]
    lo, hi = log_mag.min(), log_mag.max()
    if hi > lo:
        log_mag = (log_mag - lo) / (hi - lo)

    # 6. Log-polar resampling — maps rotation → row shift, scale → col shift
    return warp_polar(log_mag, scaling="log", order=3)


# ---------------------------------------------------------------------------
# Rotation + scale estimation
# ---------------------------------------------------------------------------

def estimate_rotation_scale(
    lsb_image: np.ndarray,
    upsample_factor: int = 10,
    key: int = 0,
    patch_size: int = 32,
    tile_mode: str = "normal",
) -> tuple[float, float]:
    """
    Estimate rotation (degrees) and scale factor between *lsb_image* and a
    tiled reference patch using Fourier-Mellin phase correlation.

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W), uint8
        LSB plane extracted from the watermarked image.
    upsample_factor : int, optional
        Sub-pixel precision factor for :func:`phase_cross_correlation`
        (default 10).
    key : int, optional
        Random seed for generating the reference patch (default 0).
    patch_size : int, optional
        Size of the reference patch in pixels (default 32).
    tile_mode : str, optional
        Tiling mode for the reference patch (default ``"normal"``).

    Returns
    -------
    angle : float
        Estimated rotation angle in degrees (counter-clockwise positive).
    scale : float
        Estimated scale factor (> 1 means the image was zoomed in).
    """
    reference = build_reference_patch(
        lsb_image.shape, patch_size=patch_size, key=key, tile_mode=tile_mode
    )

    img_lp = _fourier_mellin_spectrum(lsb_image.astype(np.float64))
    ref_lp = _fourier_mellin_spectrum(reference.astype(np.float64))

    shift, _, _ = phase_cross_correlation(
        ref_lp, img_lp,
        normalization=None,
        upsample_factor=upsample_factor,
    )

    H, W = img_lp.shape

    # Row shift → rotation angle
    angle = shift[0] / H * 360.0

    # Col shift → scale factor
    # warp_polar(scaling='log') maps column j to radius exp(j/W * log(r_max))
    r_max = np.sqrt(H**2 + W**2) / 2.0
    scale = np.exp(shift[1] / W * np.log(max(r_max, 2.0)))

    return float(angle), float(scale)


# ---------------------------------------------------------------------------
# Rotation + scale correction
# ---------------------------------------------------------------------------

def correct_rotation_scale(
    image: np.ndarray,
    angle: float,
    scale: float,
) -> np.ndarray:
    """
    Apply the inverse rotation and scale to *image*, then crop to the largest
    axis-aligned rectangle that contains no border padding.

    Parameters
    ----------
    image : ndarray of shape (H, W)
    angle : float
        Rotation angle in degrees as returned by
        :func:`estimate_rotation_scale`.
    scale : float
        Scale factor as returned by :func:`estimate_rotation_scale`.

    Returns
    -------
    corrected : ndarray of shape (H', W'), same dtype as *image*
        The crop is the largest rectangle free of border fill.
    """
    # --- inverse scale -------------------------------------------------------
    scaled = rescale(
        image, scale,
        order=0,
        preserve_range=True,
        anti_aliasing=(scale < 1.0),
        channel_axis=None,
    )

    # --- inverse rotation (expand canvas to avoid corner clipping) -----------
    rotated = rotate(
        scaled, -angle,
        resize=True,
        order=0,
        mode="constant",
        cval=0,
        preserve_range=True,
    )

    # --- largest inner rectangle after rotation ------------------------------
    # Formula from: https://stackoverflow.com/a/16778797
    h, w = scaled.shape
    a = np.deg2rad(abs(angle)) % (np.pi / 2)
    s, c = np.sin(a), np.cos(a)

    if min(h, w) <= 2 * s * c * max(h, w):
        crop_w = crop_h = int(min(h, w) / (2 * max(s, c)))
    else:
        d = c * c - s * s
        crop_w = int((w * c - h * s) / d)
        crop_h = int((h * c - w * s) / d)

    H, W = rotated.shape
    cy, cx = H // 2, W // 2

    corrected = rotated[
        cy - crop_h // 2: cy + crop_h // 2,
        cx - crop_w // 2: cx + crop_w // 2,
    ]

    return corrected.astype(image.dtype)


# ---------------------------------------------------------------------------
# Translation estimation + correction
# ---------------------------------------------------------------------------

def estimate_translation(
    lsb_image: np.ndarray,
    upsample_factor: int = 10,
    key: int = 0,
    patch_size: int = 32,
    tile_mode: str = "normal",
) -> tuple[float, float]:
    """
    Estimate sub-pixel translation via Cartesian phase correlation.

    The reference patch is regenerated at *lsb_image*'s (possibly cropped)
    shape so the comparison is valid after rotation/scale correction.

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W)
    upsample_factor : int, optional
        Sub-pixel precision factor (default 10).
    key : int, optional
        Random seed for the reference patch (default 0).
    patch_size : int, optional
        Size of the reference patch in pixels (default 32).
    tile_mode : str, optional
        Tiling mode for the reference patch (default ``"normal"``).

    Returns
    -------
    shift_row : float
    shift_col : float
    """
    reference = build_reference_patch(
        lsb_image.shape, patch_size=patch_size, key=key, tile_mode=tile_mode
    )

    shift, _, _ = phase_cross_correlation(
        reference.astype(np.float64),
        lsb_image.astype(np.float64),
        normalization=None,
        upsample_factor=upsample_factor,
    )

    return float(shift[0]), float(shift[1])


def correct_translation(
    image: np.ndarray,
    shift_row: float,
    shift_col: float,
) -> np.ndarray:
    """
    Apply a sub-pixel translation to *image* using an inverse geometric
    transform.

    Out-of-bounds pixels are filled with ``NaN`` so downstream voters can
    treat them as missing rather than as a real 0 or 1 bit.

    Parameters
    ----------
    image : ndarray of shape (H, W), values in {0, 1}
    shift_row : float
    shift_col : float

    Returns
    -------
    shifted : ndarray of shape (H, W), float32, NaN where no source pixel
    """
    # warp() applies the *inverse* mapping, so we negate the shift
    tform = SimilarityTransform(translation=(-shift_col, -shift_row))

    return warp(
        image.astype(np.float32),
        tform,
        order=0,              # nearest-neighbour — no interpolation of bits
        mode="constant",
        cval=np.nan,          # NaN padding for missing regions
        preserve_range=True,
    )


# ---------------------------------------------------------------------------
# Full synchronisation pipeline
# ---------------------------------------------------------------------------

def synchronise(
    lsb_image: np.ndarray,
    upsample_factor_rs: int = 10,
    upsample_factor_t: int = 10,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "normal",
) -> np.ndarray:
    """
    Align *lsb_image* to the tiled reference patch in two stages.

    Stage 1 — Fourier-Mellin phase correlation
        Estimates and corrects rotation + scale via log-polar phase
        correlation on the FFT magnitude spectra.

    Stage 2 — Cartesian phase correlation
        Estimates and corrects the residual translation on the
        rotation/scale-corrected image.

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W), uint8
        LSB plane extracted from the watermarked image.
    upsample_factor_rs : int, optional
        Sub-pixel precision for the rotation/scale estimate (default 10).
    upsample_factor_t : int, optional
        Sub-pixel precision for the translation estimate (default 10).
    patch_size : int, optional
        Reference patch size in pixels (default 32).
    key : int, optional
        Random seed for the reference patch (default 0).
    tile_mode : str, optional
        Tiling mode for the reference patch (default ``"normal"``).

    Returns
    -------
    aligned_lsb : ndarray of shape (H', W'), float32
        LSB plane realigned to the reference patch grid.
        NaN values indicate pixels outside the original image boundary.
    """
    # Stage 1 — rotation + scale
    angle, scale = estimate_rotation_scale(
        lsb_image,
        upsample_factor=upsample_factor_rs,
        key=key,
        patch_size=patch_size,
        tile_mode=tile_mode,
    )
    print(f"Estimated rotation: {angle:.2f}°, scale: {scale:.4f}")
    lsb_rs = correct_rotation_scale(lsb_image, angle, scale)

    # Stage 2 — translation (reference is regenerated at the new crop shape)
    dr, dc = estimate_translation(
        lsb_rs,
        upsample_factor=upsample_factor_t,
        key=key,
        patch_size=patch_size,
        tile_mode=tile_mode,
    )
    return correct_translation(lsb_rs, dr, dc)
