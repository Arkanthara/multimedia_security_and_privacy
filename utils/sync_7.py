"""
utils/sync.py
=============
Frequency-domain synchronisation for **spatial** watermark decoding.

Unlike the LSB variant, the input here is a floating-point *residual*
(watermarked_image − denoised_image) whose watermark signal is bipolar
(±alpha), not binary {0, 1}.  The reference is therefore built with
:func:`~utils.patch.build_reference_patch_bipolar` (values in {-1, +1}).

Pipeline
--------
1. Spatial autocorrelation via Wiener-Khinchin (ifft(F·F*)) on the residual
   → isolates the periodic tiling structure regardless of image content.
2. Log-polar phase correlation  → estimate rotation + scale.
3. Correct rotation / scale     → cropped & NaN-padded (no resize artefacts).
4. Cartesian phase correlation  → estimate translation.
5. Correct translation          → final aligned residual (float32, NaN padding).

The output is a float32 residual ready to be passed to
:func:`~utils.patch.extract_bits_from_spatial`.
"""

import numpy as np
from skimage.transform import rescale, rotate, warp, SimilarityTransform, warp_polar
from skimage.registration import phase_cross_correlation

from utils.patch import build_reference_patch_bipolar


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _autocorr_spectrum(image: np.ndarray) -> np.ndarray:
    """
    Spatial autocorrelation of *image* via the Wiener-Khinchin theorem.

    ``autocorr = ifft2(F · conj(F))``, log-scaled and DC-centred.

    NaN values (from previous alignment stages) are replaced with 0 before
    the FFT so they do not pollute the spectrum.

    Parameters
    ----------
    image : ndarray of shape (H, W), real-valued (float)

    Returns
    -------
    autocorr : ndarray of shape (H, W), dtype float64
        Log-scaled spatial autocorrelation, DC at centre.
    """
    img = np.nan_to_num(image, nan=0.0).astype(np.float64)
    img = img - img.mean()
    f   = np.fft.fft2(img)
    ac  = np.fft.ifft2(f * np.conj(f)).real
    ac  = np.fft.fftshift(ac)
    return np.log1p(np.maximum(ac, 0.0))


def _to_log_polar(spectrum: np.ndarray) -> np.ndarray:
    """
    Convert a centred 2-D spectrum to log-polar coordinates.

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
    residual: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    upsample_factor: int = 10,
    upsample_factor_patch: int = 2,
) -> tuple[float, float]:
    """
    Estimate rotation (degrees) and scale factor between *residual* and
    the bipolar reference patch using log-polar phase correlation on their
    spatial autocorrelations.

    Parameters
    ----------
    residual : ndarray of shape (H, W), float
        Spatial residual (watermarked − denoised).
    patch_size : int
        Logical patch side-length (before upsampling).  Must match encode.
    key : int
        Watermark key used to rebuild the reference patch.
    tile_mode : {"normal", "symmetric"}
        Tiling mode — must match encode.
    upsample_factor : int
        Sub-pixel precision for phase correlation (higher = finer, slower).
    upsample_factor_patch : int
        Upsampling factor used when the watermark was embedded.

    Returns
    -------
    angle : float
        Estimated rotation angle in degrees (counter-clockwise).
    scale : float
        Estimated scale factor (> 1 means image was zoomed in).
    """
    reference = build_reference_patch_bipolar(
        residual.shape, patch_size=patch_size, key=key,
        tile_mode=tile_mode, upsample_factor=upsample_factor_patch,
    )

    res_lp = _to_log_polar(_autocorr_spectrum(residual))
    ref_lp = _to_log_polar(_autocorr_spectrum(reference))

    shift, _, _ = phase_cross_correlation(
        ref_lp, res_lp,
        normalization=None,
        upsample_factor=upsample_factor,
    )

    H, W  = res_lp.shape
    angle = shift[0] / H * 360.0

    output_radius = np.sqrt(H ** 2 + W ** 2) / 2.0
    scale = np.exp(shift[1] / W * np.log(max(output_radius, 2.0)))

    return float(angle), float(scale)


# ---------------------------------------------------------------------------
# Rotation + scale correction
# ---------------------------------------------------------------------------

def correct_rotation_scale(
    residual: np.ndarray,
    angle: float,
    scale: float,
) -> np.ndarray:
    """
    Apply inverse rotation and scale to *residual*.

    Out-of-bounds pixels are filled with NaN so downstream tile-voting can
    ignore them correctly.  The image is cropped to the largest axis-aligned
    rectangle inside the rotated frame (no black borders).

    Parameters
    ----------
    residual : ndarray of shape (H, W), float
    angle : float
        Rotation angle in degrees (as returned by
        :func:`estimate_rotation_scale`).
    scale : float
        Scale factor.

    Returns
    -------
    corrected : ndarray of shape (H', W'), float32
        Cropped, aligned residual; H' ≤ H, W' ≤ W.
    """
    # Replace pre-existing NaNs with 0 for geometric transforms (they are
    # re-introduced via cval=nan below).
    src = np.nan_to_num(residual, nan=0.0).astype(np.float64)

    scaled = rescale(
        src, scale, order=1,
        preserve_range=True, anti_aliasing=False, channel_axis=None,
        cval=np.nan,
    )

    rotated = rotate(
        scaled, -angle, resize=True,
        order=1, mode="constant", cval=np.nan, preserve_range=True,
    )

    # Compute largest inscribed rectangle after rotation
    h, w = scaled.shape
    a    = np.deg2rad(abs(angle)) % (np.pi / 2)
    s, c = np.sin(a), np.cos(a)

    if min(h, w) <= 2 * s * c * max(h, w):
        crop_w = crop_h = int(min(h, w) / (2 * max(s, c) + 1e-8))
    else:
        d      = c * c - s * s
        crop_w = int((w * c - h * s) / (d + 1e-8))
        crop_h = int((h * c - w * s) / (d + 1e-8))

    crop_h = max(crop_h, 1)
    crop_w = max(crop_w, 1)

    H, W = rotated.shape
    cy, cx = H // 2, W // 2

    return rotated[
        cy - crop_h // 2: cy + crop_h // 2,
        cx - crop_w // 2: cx + crop_w // 2,
    ].astype(np.float32)


# ---------------------------------------------------------------------------
# Translation estimation + correction
# ---------------------------------------------------------------------------

def estimate_translation(
    residual: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    upsample_factor: int = 10,
    upsample_factor_patch: int = 2,
) -> tuple[float, float]:
    """
    Estimate sub-pixel translation via Cartesian phase correlation between
    *residual* and the bipolar reference patch.

    Parameters
    ----------
    residual : ndarray of shape (H, W), float
    patch_size : int
    key : int
    tile_mode : {"normal", "symmetric"}
    upsample_factor : int
        Sub-pixel precision factor.
    upsample_factor_patch : int
        Upsampling factor used when the watermark was embedded.

    Returns
    -------
    shift_row : float
    shift_col : float
    """
    reference = build_reference_patch_bipolar(
        residual.shape, patch_size=patch_size, key=key,
        tile_mode=tile_mode, upsample_factor=upsample_factor_patch,
    )

    res_f = np.nan_to_num(residual, nan=0.0).astype(np.float64)

    shift, _, _ = phase_cross_correlation(
        reference.astype(np.float64),
        res_f,
        normalization=None,
        upsample_factor=upsample_factor,
    )
    return float(shift[0]), float(shift[1])


def correct_translation(
    residual: np.ndarray,
    shift_row: float,
    shift_col: float,
) -> np.ndarray:
    """
    Apply inverse translation to *residual* using a geometric warp.

    Out-of-bounds areas are filled with NaN so tile-voting ignores them.
    No value remapping is applied — float residuals pass through unchanged.

    Parameters
    ----------
    residual : ndarray of shape (H, W), float
    shift_row : float
    shift_col : float

    Returns
    -------
    shifted : ndarray of shape (H, W), float32 with NaNs in padded regions
    """
    # warp applies the *inverse* transform, so we negate the shift
    tform = SimilarityTransform(translation=(-shift_col, -shift_row))

    shifted = warp(
        residual.astype(np.float32),
        tform,
        order=1,                # bilinear — appropriate for float residuals
        mode="constant",
        cval=np.nan,
        preserve_range=True,
    )
    return shifted


# ---------------------------------------------------------------------------
# Full synchronisation pipeline
# ---------------------------------------------------------------------------

def synchronise(
    residual: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    upsample_factor_rs: int = 10,
    upsample_factor_t: int = 10,
    upsample_factor_patch: int = 2,
) -> np.ndarray:
    """
    Align *residual* to the bipolar reference patch grid using a two-stage
    frequency-domain pipeline.

    Stage 1 — Log-polar phase correlation on spatial autocorrelations
        → rotation + scale correction (NaN-padded crop).
    Stage 2 — Cartesian phase correlation on corrected residual
        → translation correction (NaN-padded shift).

    Parameters
    ----------
    residual : ndarray of shape (H, W), float
        Per-pixel residual (watermarked_image − denoised_image).
    patch_size : int
        Logical patch side-length used at embed time.
    key : int
        Watermark key used to rebuild the reference.
    tile_mode : {"normal", "symmetric"}
        Tiling mode used at embed time.
    upsample_factor_rs : int
        Sub-pixel precision for rotation/scale phase correlation.
    upsample_factor_t : int
        Sub-pixel precision for translation phase correlation.
    upsample_factor_patch : int
        Pixel-block size used at embed time (upsample_factor in patch.py).

    Returns
    -------
    aligned : ndarray of shape (H', W'), float32
        Residual realigned to the reference patch grid, NaNs where data
        were unavailable after geometric correction.  Ready for
        :func:`~utils.patch.extract_bits_from_spatial`.
    """
    # Stage 1 — rotation + scale
    angle, scale = estimate_rotation_scale(
        residual,
        patch_size=patch_size,
        key=key,
        tile_mode=tile_mode,
        upsample_factor=upsample_factor_rs,
        upsample_factor_patch=upsample_factor_patch,
    )
    res_rs = correct_rotation_scale(residual, angle, scale)

    # Stage 2 — translation
    dr, dc = estimate_translation(
        res_rs,
        patch_size=patch_size,
        key=key,
        tile_mode=tile_mode,
        upsample_factor=upsample_factor_t,
        upsample_factor_patch=upsample_factor_patch,
    )
    aligned = correct_translation(res_rs, dr, dc)

    return aligned
