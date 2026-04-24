"""
utils/sync.py
=============
Frequency-domain synchronisation for watermark decoding.

Pipeline
--------
1. Spatial autocorrelation via Wiener-Khinchin on both image and reference
2. Non-maximum suppression (NMS) + thresholding + binarization to isolate lattice peaks
3. Match image peaks to reference peaks → estimate affine transform
4. Apply inverse affine (crop, no padding artefacts)
5. Sum all patch-sized blocks into one block (symmetric-aware)
6. Correlate summed block against reference patch → estimate translation
7. Apply circular shift to align block with reference
8. Return aligned, summed block of shape (patch_size * upsample_factor,
   patch_size * upsample_factor)

The ``upsample_factor`` parameter must match the value used in patch.py at
encode time.  It controls the pixel-block size: each logical patch cell is
represented by ``upsample_factor × upsample_factor`` image pixels.
"""

import numpy as np
import cv2
from scipy.ndimage import maximum_filter
from utils.patch import build_reference_patch, generate_base_patch, upsample_patch


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation_fft(image: np.ndarray) -> np.ndarray:
    """
    Compute the spatial autocorrelation via the Wiener-Khinchin theorem.

    ``autocorr = ifft2(F · conj(F))``, shifted so DC is at the centre.

    The input is expected to already have its mean removed (DC = 0) so that
    the central peak does not dominate.  No additional min-max normalisation
    is applied here to avoid fighting the caller's DC removal.

    Parameters
    ----------
    image : ndarray of shape (H, W), real-valued, zero-mean

    Returns
    -------
    autocorr : ndarray of shape (H, W), dtype float64
    """
    # FIX 1 – removed the internal min-max normalisation that was undoing the
    # mean-subtraction performed in `synchronise`.  The caller already removes
    # the DC component; a second rescaling scrambles the zero-mean property.
    f = np.fft.fft2(image.astype(np.float64))
    return np.fft.fftshift(np.real(np.fft.ifft2(f * np.conj(f))))


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def non_maximum_suppression(image: np.ndarray, size: int = 31) -> np.ndarray:
    """
    Retain only local maxima within a ``size × size`` neighbourhood.

    Parameters
    ----------
    image : ndarray of shape (H, W)
    size  : int

    Returns
    -------
    nms : ndarray of shape (H, W)
    """
    return (image == maximum_filter(image, size=size)) * image


# ---------------------------------------------------------------------------
# Thresholding + binarisation  (NEW)
# ---------------------------------------------------------------------------

def threshold_and_binarize(nms: np.ndarray, k: float = 3.0) -> np.ndarray:
    """
    Keep only statistically significant NMS peaks and binarize them.

    Steps
    -----
    1. Consider only strictly positive NMS values (actual local maxima).
    2. Compute their mean ``μ`` and standard deviation ``σ``.
    3. Threshold: keep peaks with value > μ + k·σ.
    4. Binarize: set surviving peaks to 1.

    Why this matters
    ----------------
    Raw NMS returns *every* local maximum, including hundreds of noise bumps.
    Without thresholding, ``detect_interest_points`` anchors on a noise peak
    instead of a true lattice peak.  Binarizing ensures that peak amplitude
    does not bias the proximity chain (all genuine lattice peaks should look
    equally strong after autocorrelation).

    Parameters
    ----------
    nms : ndarray of shape (H, W) – output of non_maximum_suppression
    k   : float – number of standard deviations above the mean

    Returns
    -------
    binary : ndarray of shape (H, W), dtype bool  (True at surviving peaks)
    """
    positive = nms[nms > 0]
    if positive.size == 0:
        return np.zeros_like(nms, dtype=bool)
    threshold = positive.mean() + k * positive.std()
    return nms > threshold


# ---------------------------------------------------------------------------
# Interest point detection
# ---------------------------------------------------------------------------

def detect_interest_points(binary_peaks: np.ndarray,
                           min_angle_deg: float = 20.0) -> np.ndarray:
    """
    Extract a quadrilateral cell from a lattice using a simple geometric rule.

    Strategy
    --------
    p1 : center (closest peak to image center)
    p2 : closest neighbor to p1
    p3 : next closest point with a different direction from p2
         (angle > min_angle and < 180 - min_angle)
    p4 : real point closest to (p2 + p3)

    The returned order is: [p1, p2, p4, p3].

    Parameters
    ----------
    binary_peaks : ndarray (H, W)
        Binary map of detected peaks.
    min_angle_deg : float
        Minimum angle between p2 and p3 (in degrees).

    Returns
    -------
    points : ndarray (4, 2)
        Quadrilateral in centered coordinates.

    Raises
    ------
    ValueError if a valid configuration cannot be found.
    """
    H, W = binary_peaks.shape

    ys, xs = np.nonzero(binary_peaks)
    if len(ys) < 4:
        raise ValueError("Not enough peaks.")

    coords = np.stack([ys, xs], axis=1).astype(np.float64)

    # --- p1: center ---
    center_idx = np.argmin(np.linalg.norm(coords - [H/2, W/2], axis=1))
    center = coords[center_idx]

    centered = coords - center
    others = centered[np.arange(len(centered)) != center_idx]

    # sort by distance to p1
    dists = np.linalg.norm(others, axis=1)
    order = np.argsort(dists)
    others = others[order]

    p1 = np.zeros(2)

    # --- p2: closest ---
    p2 = others[0]

    # --- p3: non-colinear ---
    min_angle = np.radians(min_angle_deg)
    p3 = None

    for p in others[1:]:
        cos = np.dot(p, p2) / (np.linalg.norm(p) * np.linalg.norm(p2) + 1e-8)
        angle = np.arccos(np.clip(cos, -1, 1))

        if min_angle < angle < (np.pi - min_angle):
            p3 = p
            break

    if p3 is None:
        raise ValueError("No valid p3 found.")

    # --- p4: closest to p2 + p3 ---
    # target = p2 + p3

    # mask = ~(
    #     (np.linalg.norm(others - p2, axis=1) < 1e-6) |
    #     (np.linalg.norm(others - p3, axis=1) < 1e-6)
    # )

    # candidates = others[mask]
    # if len(candidates) == 0:
    #     raise ValueError("No candidates for p4.")

    # p4 = candidates[np.argmin(np.linalg.norm(candidates - target, axis=1))]

    # result = np.stack([p1, p2, p4, p3])
    result = np.stack([p1, p2, p3])  # FIX: removed p4 to handle cases where it can't be found
    result += center  # convert back to image coordinates
    return result.astype(np.float64)


# ---------------------------------------------------------------------------
# Affine estimation
# ---------------------------------------------------------------------------

def estimate_affine(img_ac: np.ndarray, ref_ac: np.ndarray, nms_size: int = 31) -> np.ndarray:
    """
    Estimate the 2-D affine matrix mapping image lattice peaks to reference peaks.

    Interest points are detected independently in each autocorrelation map,
    then matched by position in the proximity chain (p1↔p1, p2↔p2, p3↔p3).

    Parameters
    ----------
    img_ac   : ndarray of shape (H, W)
    ref_ac   : ndarray of shape (H, W)
    nms_size : int

    Returns
    -------
    M : ndarray of shape (2, 3) — OpenCV affine matrix
    """
    img_pts = detect_interest_points(non_maximum_suppression(img_ac, nms_size))
    ref_pts = detect_interest_points(non_maximum_suppression(ref_ac, nms_size))

    # cv2.getAffineTransform expects (x, y) = (col, row)
    src = img_pts[:, ::-1].astype(np.float32)
    dst = ref_pts[:, ::-1].astype(np.float32)
    return cv2.getAffineTransform(src, dst)


# ---------------------------------------------------------------------------
# Affine correction
# ---------------------------------------------------------------------------

def correct_affine(
    image: np.ndarray,
    M: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Apply an affine warp and crop to the largest centred rectangle with no
    padding artefacts.

    Parameters
    ----------
    image         : ndarray of shape (H, W) — uint8 (LSB) or float32 (residual)
    M             : ndarray of shape (2, 3)
    interpolation : int — OpenCV interpolation flag

    Returns
    -------
    cropped : ndarray, same dtype as image
    """
    H, W = image.shape
    warped = cv2.warpAffine(
        image.astype(np.float32), M, (W, H),
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    angle = abs(np.arctan2(M[1, 0], M[0, 0])) % (np.pi / 2)
    s, c = np.sin(angle), np.cos(angle)

    if min(H, W) <= 2 * s * c * max(H, W):
        crop_h = crop_w = int(min(H, W) / (2 * max(s, c) + 1e-9))
    else:
        d = c * c - s * s + 1e-9
        crop_h = int((H * c - W * s) / d)
        crop_w = int((W * c - H * s) / d)

    cy, cx = H // 2, W // 2
    crop_h, crop_w = max(crop_h, 1), max(crop_w, 1)
    return warped[
        cy - crop_h // 2 : cy + crop_h // 2,
        cx - crop_w // 2 : cx + crop_w // 2,
    ].astype(image.dtype)


# ---------------------------------------------------------------------------
# Block summation
# ---------------------------------------------------------------------------

def sum_blocks(
    image: np.ndarray,
    patch_size: int,
    tile_mode: str = "normal",
    upsample_factor: int = 2,
) -> np.ndarray:
    """
    Divide image into non-overlapping blocks and sum them into a single block.

    When ``tile_mode == "symmetric"``, every other tile is flipped before
    summation to undo the symmetric tiling applied during encoding.

    Parameters
    ----------
    image           : ndarray of shape (H, W)
    patch_size      : int — logical patch side length (before upsampling)
    tile_mode       : {"normal", "symmetric"}
    upsample_factor : int — must match the value used during encoding

    Returns
    -------
    block : ndarray of shape (up, up), dtype float64
        where ``up = patch_size * upsample_factor``
    """
    up = patch_size * upsample_factor
    H, W = image.shape

    img = np.pad(
        image.astype(np.float64),
        ((0, (-H) % up), (0, (-W) % up)),
        mode="constant",
    )
    H2, W2 = img.shape
    tiles = img.reshape(H2 // up, up, W2 // up, up).transpose(1, 3, 0, 2)  # (up, up, Ty, Tx)

    if tile_mode == "symmetric":
        tiles[:, :, 1::2, :] = tiles[::-1, :, 1::2, :]
        tiles[:, :, :, 1::2] = tiles[:, ::-1, :, 1::2]

    return np.sum(tiles, axis=(2, 3))


# ---------------------------------------------------------------------------
# Translation via correlation on summed block
# ---------------------------------------------------------------------------

def estimate_translation_block(
    summed_block: np.ndarray,
    patch_size: int,
    key: int,
    nms_size: int = 5,
    upsample_factor: int = 2,
) -> tuple[int, int]:
    """
    Estimate integer translation between summed_block and the reference patch
    via cross-correlation + NMS.

    Parameters
    ----------
    summed_block    : ndarray of shape (up, up)
    patch_size      : int
    key             : int
    nms_size        : int
    upsample_factor : int

    Returns
    -------
    shift_row, shift_col : int
    """
    ref_up = upsample_patch(generate_base_patch(patch_size, key),
                            factor=upsample_factor).astype(np.float64)

    corr = np.fft.fftshift(np.real(np.fft.ifft2(
        np.fft.fft2(summed_block.astype(np.float64)) * np.conj(np.fft.fft2(ref_up))
    )))

    peak_y, peak_x = np.unravel_index(
        np.argmax(non_maximum_suppression(corr, size=nms_size)), corr.shape
    )
    cy, cx = corr.shape[0] // 2, corr.shape[1] // 2
    return int(peak_y) - cy, int(peak_x) - cx


def correct_translation_block(
    summed_block: np.ndarray, shift_row: int, shift_col: int
) -> np.ndarray:
    """
    Apply a circular shift to summed_block to align it with the reference.

    Parameters
    ----------
    summed_block       : ndarray of shape (up, up)
    shift_row, shift_col : int

    Returns
    -------
    aligned : ndarray of shape (up, up)
    """
    return np.roll(summed_block, (-shift_row, -shift_col), axis=(0, 1))


# ---------------------------------------------------------------------------
# Full synchronisation pipeline
# ---------------------------------------------------------------------------

def synchronise(
    img: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    nms_size_ac: int = 31,
    nms_size_corr: int = 5,
    upsample_factor: int = 2,
) -> np.ndarray:
    """
    Align img to the reference watermark grid and return the summed,
    translation-corrected block.

    Pipeline
    --------
    1. Compute autocorrelations of image and reference via FFT.
    2. Detect three lattice peaks in each → estimate affine transform.
    3. Warp image with affine + safe centre-crop.
    4. Sum all (patch_size * upsample_factor)² blocks of the corrected image.
    5. Cross-correlate summed block with reference patch → translation.
    6. Circular-shift to align.

    Parameters
    ----------
    img            : ndarray of shape (H, W) — uint8 {0,1} or float32 residual
    patch_size     : int
    key            : int
    tile_mode      : {"normal", "symmetric"}
    nms_size_ac    : int — NMS window for autocorrelation peak detection
    nms_size_corr  : int — NMS window for correlation peak detection
    upsample_factor: int

    Returns
    -------
    aligned_block : ndarray of shape (patch_size * upsample_factor,
                                      patch_size * upsample_factor), float64
    """
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 else cv2.INTER_LINEAR

    reference = build_reference_patch(
        img.shape, patch_size=patch_size, key=key,
        tile_mode=tile_mode, upsample_factor=upsample_factor,
    )

    img_ac = autocorrelation_fft(img.astype(np.float64) - img.mean())
    ref_ac = autocorrelation_fft(reference.astype(np.float64) - reference.mean())

    M = estimate_affine(img_ac, ref_ac, nms_size=nms_size_ac)
    img_corrected = correct_affine(img, M, interpolation=interp)

    summed = sum_blocks(img_corrected, patch_size=patch_size,
                        tile_mode=tile_mode, upsample_factor=upsample_factor)
    shift_row, shift_col = estimate_translation_block(
        summed, patch_size=patch_size, key=key,
        nms_size=nms_size_corr, upsample_factor=upsample_factor,
    )
    return correct_translation_block(summed, shift_row, shift_col)