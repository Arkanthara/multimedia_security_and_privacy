"""
utils/sync.py
=============
Frequency-domain synchronisation for watermark decoding.

Pipeline
--------
1. Spatial autocorrelation via Wiener-Khinchin on both image and reference
2. Non-maximum suppression (NMS) to isolate lattice peaks
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

    Parameters
    ----------
    image : ndarray of shape (H, W), real-valued

    Returns
    -------
    autocorr : ndarray of shape (H, W), dtype float64
    """
    normalized = (image - image.min()) / (image.max() - image.min() + 1e-9)
    f = np.fft.fft2(normalized.astype(np.float64))
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
# Interest point detection
# ---------------------------------------------------------------------------

def detect_interest_points(nms: np.ndarray) -> np.ndarray:
    """
    Select three interest points from NMS peaks via a proximity chain.

    Steps
    -----
    1. Find the center peak: NMS peak closest to the image center.
    2. Center all peak coordinates relative to this peak (center → origin).
    3. p1 = center peak = (0, 0).
    4. p2 = non-center peak closest to p1.
    5. p3 = non-center peak closest to p2.

    Parameters
    ----------
    nms : ndarray of shape (H, W)

    Returns
    -------
    points : ndarray of shape (3, 2)
        [p1, p2, p3] as (row, col) in centered coordinates.
    """
    H, W = nms.shape
    ys, xs = np.nonzero(nms)
    coords = np.stack([ys, xs], axis=1).astype(np.float64)  # (N, 2)

    # Center peak: NMS peak closest to the image center
    center_idx = np.argmin(np.linalg.norm(coords - [H / 2.0, W / 2.0], axis=1))
    centered = coords - coords[center_idx]              # center peak → (0, 0)

    # Non-center peaks
    others = centered[np.arange(len(centered)) != center_idx]

    p1 = np.zeros(2)                                            # (0, 0)
    p2 = others[np.argmin(np.linalg.norm(others - p1, axis=1))]
    p3 = others[np.argmin(np.linalg.norm(others - p2, axis=1))]

    return np.stack([p1, p2, p3])


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
