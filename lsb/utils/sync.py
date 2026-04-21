"""
utils/sync.py
=============
Frequency-domain synchronisation for LSB watermark decoding.

Pipeline
--------
1. Spatial autocorrelation via Wiener-Khinchin (ifft(F·F*))
2. Log-polar phase correlation  → estimate rotation + scale
3. Correct rotation / scale     → cropped (no resize artefacts)
4. Cartesian phase correlation  → estimate translation
5. Correct translation          → final aligned image
"""

import numpy as np
from skimage.transform import warp, AffineTransform, warp_polar
from skimage.registration import phase_cross_correlation


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _autocorr_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Compute the spatial autocorrelation of *image* via the Wiener-Khinchin
    theorem: ``autocorr = ifft2(F · conj(F))``, shifted to centre.

    The result captures the periodic structure of the watermark tiling
    and is rotation-equivariant, making it suitable for log-polar
    phase correlation.

    Parameters
    ----------
    image : ndarray of shape (H, W), real-valued

    Returns
    -------
    autocorr : ndarray of shape (H, W), dtype float64
        Log-scaled spatial autocorrelation, DC at centre.
    """
    # Mean-center so non-period autocorrelation lags collapse to ≈ 0,
    # leaving clear peaks only at tile-period offsets.
    img     = image - image.mean()
    f       = np.fft.fft2(img)
    # Multiply by conjugate — keep complex product, take real part after ifft
    ac      = np.fft.ifft2(f * np.conj(f)).real
    ac      = np.fft.fftshift(ac)
    # Log-scale for phase correlation; clip negatives (numerical noise)
    return np.log1p(np.maximum(ac, 0.0))


def _to_log_polar(spectrum: np.ndarray) -> np.ndarray:
    """
    Convert a centred 2-D representation to log-polar coordinates.

    Parameters
    ----------
    spectrum : ndarray of shape (H, W)

    Returns
    -------
    log_polar : ndarray of shape (H, W)
    """
    return warp_polar(spectrum, scaling="log")


# ---------------------------------------------------------------------------
# Rotation + scale estimation
# ---------------------------------------------------------------------------

def estimate_rotation_scale(
    lsb_image: np.ndarray,
    reference: np.ndarray,
    upsample_factor: int = 1,
) -> tuple[float, float]:
    """
    Estimate rotation (degrees) and scale factor between *lsb_image* and
    *reference* using log-polar phase correlation on the spatial
    autocorrelations.

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W), uint8
        LSB plane extracted from the watermarked image.
    reference : ndarray of shape (H, W), uint8
        Reference patch tiled to the same size.
    upsample_factor : int, optional
        Sub-pixel precision factor passed to :func:`phase_cross_correlation`
        (default 1 = pixel precision, higher = more precise but slower).

    Returns
    -------
    angle : float
        Estimated rotation angle in degrees (counter-clockwise).
    scale : float
        Estimated scale factor (> 1 means image was zoomed in).
    """
    # Use raw float64 — values stay in {0.0, 1.0}, not divided by 255
    img_f  = lsb_image.astype(np.float64)
    ref_f  = reference.astype(np.float64)

    img_lp = _to_log_polar(_autocorr_spectrum(img_f))
    ref_lp = _to_log_polar(_autocorr_spectrum(ref_f))

    shift, _, _ = phase_cross_correlation(
        ref_lp, img_lp,
        normalization=None,
        upsample_factor=upsample_factor,
    )

    H, W   = img_lp.shape
    angle  = shift[0] / H * 360.0

    # warp_polar(scaling='log') maps cols to natural-log-spaced radii
    # r_j = exp(j / W * log(output_radius)).  Recover scale from col shift.
    output_radius = np.sqrt(H ** 2 + W ** 2) / 2.0
    scale = np.exp(shift[1] / W * np.log(max(output_radius, 2.0)))

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
    Apply inverse rotation and scale to *image*.

    The image is cropped to its original size — no resize artefacts.

    Parameters
    ----------
    image : ndarray of shape (H, W)
    angle : float
        Rotation angle in degrees (as returned by
        :func:`estimate_rotation_scale`).
    scale : float
        Scale factor.

    Returns
    -------
    corrected : ndarray of shape (H, W), dtype uint8
    """
    H, W = image.shape
    cx, cy = W / 2.0, H / 2.0

    # Inverse transform directly (clearer than inverting later)
    tform = (
        AffineTransform(translation=(-cx, -cy))
        + AffineTransform(
            rotation=np.deg2rad(-angle),   # invert rotation
            scale=(1.0 / scale, 1.0 / scale),  # invert scale
        )
        + AffineTransform(translation=(cx, cy))
    )

    corrected = warp(
        image.astype(np.float32),
        tform,
        preserve_range=True,
        order=0,          # IMPORTANT for binary / label images
        mode="constant",
        cval=0,
    )

    return corrected.astype(image.dtype)


# ---------------------------------------------------------------------------
# Translation estimation + correction
# ---------------------------------------------------------------------------

def estimate_translation(
    lsb_image: np.ndarray,
    reference: np.ndarray,
    upsample_factor: int = 10,
) -> tuple[float, float]:
    """
    Estimate sub-pixel translation via Cartesian phase correlation.

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W)
    reference : ndarray of shape (H, W)
    upsample_factor : int, optional
        Sub-pixel precision factor (default 10; use higher for finer shifts).

    Returns
    -------
    shift_row : float
    shift_col : float
    """
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
    Shift *image* by (-shift_row, -shift_col), wrapping with ``np.roll``.

    Parameters
    ----------
    image : ndarray of shape (H, W)
    shift_row : float
    shift_col : float

    Returns
    -------
    shifted : ndarray of shape (H, W), same dtype
    """
    return np.roll(
        image,
        (int(round(-shift_row)), int(round(-shift_col))),
        axis=(0, 1),
    )


# ---------------------------------------------------------------------------
# Full synchronisation pipeline
# ---------------------------------------------------------------------------

def synchronise(
    lsb_image: np.ndarray,
    reference: np.ndarray,
    upsample_factor_rs: int = 1,
    upsample_factor_t: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align *lsb_image* to *reference* using a two-stage pipeline.

    Stage 1 — Log-polar phase correlation on spatial autocorrelations
        → rotation + scale correction.
    Stage 2 — Cartesian phase correlation on corrected image
        → translation correction.

    Parameters
    ----------
    lsb_image : ndarray of shape (H, W), uint8
    reference : ndarray of shape (H, W), uint8
        Reference patch tiled to the same size as *lsb_image*.
    upsample_factor_rs : int, optional
        Sub-pixel factor for rotation/scale phase correlation (default 1).
    upsample_factor_t : int, optional
        Sub-pixel factor for translation phase correlation (default 10).

    Returns
    -------
    aligned_lsb : ndarray of shape (H, W), uint8
        LSB plane realigned to the reference patch grid.
    aligned_reference : ndarray of shape (H, W), uint8
        Same as *reference* (returned for API consistency).
    """
    # Stage 1 — rotation + scale
    angle, scale = estimate_rotation_scale(
        lsb_image, reference, upsample_factor=upsample_factor_rs
    )
    print(f"Estimated rotation: {angle}°, scale: {scale}")
    lsb_rs = correct_rotation_scale(lsb_image, angle, scale)

    # Stage 2 — translation
    dr, dc    = estimate_translation(lsb_rs, reference, upsample_factor=upsample_factor_t)
    lsb_final = correct_translation(lsb_rs, dr, dc)

    return lsb_final, reference
