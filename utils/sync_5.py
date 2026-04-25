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
from scipy.ndimage import gaussian_filter, maximum_filter
from utils.patch import build_reference_patch, generate_base_patch, upsample_patch
from skimage.transform import warp, SimilarityTransform


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def correlation_fft(img_1: np.ndarray, img_2: np.ndarray) -> np.ndarray:
    """
    Compute the cross-correlation between two images via the Wiener-Khinchin theorem.

    ``crosscorr = ifft2(F1 · conj(F2))``, shifted so DC is at the centre.

    The input is expected to already have its mean removed (DC = 0) so that
    the central peak does not dominate.  No additional min-max normalisation
    is applied here to avoid fighting the caller's DC removal.

    Parameters
    ----------
    img_1 : ndarray of shape (H, W), real-valued, zero-mean
    img_2 : ndarray of shape (h, w), real-valued, zero-mean

    Returns
    -------
    crosscorr : ndarray of shape (H, W), dtype float64
    """
    # FIX 1 – removed the internal min-max normalisation that was undoing the
    # mean-subtraction performed in `synchronise`.  The caller already removes
    # the DC component; a second rescaling scrambles the zero-mean property.
    f1 = img_1.astype(np.float64) - img_1.mean()
    f2 = img_2.astype(np.float64) - img_2.mean()
    F1 = np.fft.fft2(f1)
    F2 = np.fft.fft2(f2, s=f1.shape)
    return np.fft.fftshift(np.real(np.fft.ifft2(F1 * np.conj(F2))))


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

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
    blurred = gaussian_filter(image, sigma=0.5)
    return (blurred == maximum_filter(blurred, size=size)) * image  # original values
 
 
# ---------------------------------------------------------------------------
# Thresholding + binarisation  (NEW)
# ---------------------------------------------------------------------------
 
def select_lattice_peaks(ac: np.ndarray, n_peaks: int = 15) -> np.ndarray:
    """
    Select strongest peaks in autocorrelation map.

    Parameters
    ----------
    ac : ndarray (H, W)
        Autocorrelation map already processed by NMS.
    n_peaks : int
        Number of peaks to keep.

    Returns
    -------
    peaks : ndarray (H, W) with the k strongest peaks
    """
    flat = ac.ravel()

    # Find the threshold value (k-th largest)
    thresh = np.partition(flat, -n_peaks)[-n_peaks]

    # Keep only values >= threshold
    result = np.where(ac >= thresh, ac, 0)

    return result


# ---------------------------------------------------------------------------
# Interest point detection
# ---------------------------------------------------------------------------

def expand_peaks(image , expansion_size=3):
    expanded = np.zeros_like(image)
    peaks = np.argwhere(image > 0)
    for peak in peaks:
        x, y = peak
        expanded[ max (0, x-expansion_size): min (image.shape[0], x+expansion_size+1), max (0, y-expansion_size): min (image.shape[1], y+expansion_size+1)] = 1
    return expanded

def get_center_peak(peaks: np.ndarray) -> np.ndarray:
    """
    Find the peak closest to the center of the image.

    Parameters
    ----------
    peaks : ndarray (H, W)
        Map of detected peaks.

    Returns
    -------
    center : ndarray (2,)
        Coordinates of the center peak.
    """
    H, W = peaks.shape
    ys, xs = np.nonzero(peaks)
    coords = np.stack((ys, xs), axis=1).astype(float)
    center = coords[np.argmin(np.linalg.norm(coords - [H // 2, W // 2], axis=1))]
    return center

def detect_interest_points(peaks: np.ndarray, min_angle_deg: float = 20.0) -> np.ndarray:
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
    peaks : ndarray (H, W)
        Map of detected peaks.
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
    H, W = peaks.shape
    ys, xs = np.nonzero(peaks)
    if len(ys) < 3:
        raise ValueError("Not enough peaks.")

    coords = np.stack((ys, xs), axis=1).astype(float)
    center = get_center_peak(peaks)

    others = coords - center
    others = others[(others != 0).any(axis=1)]

    dists = np.linalg.norm(others, axis=1)
    others = others[np.argsort(dists)]   # sort by distance, not by score

    # p2 = nearest neighbour
    p2 = others[0]

    # p3 = nearest point with sufficient angular separation from p2
    cos    = (others @ p2) / (np.linalg.norm(others, axis=1) * np.linalg.norm(p2) + 1e-8)
    angles = np.arccos(np.clip(cos, -1, 1))
    tol    = np.radians(min_angle_deg)
    valid  = (angles > tol) & (angles < np.pi - tol)
    valid[0] = False

    if not np.any(valid):
        raise ValueError("All strong peaks are collinear.")

    # pick the closest valid one (not the highest-scored)
    p3_idx = np.argmax(valid)   # first True = smallest distance among valid
    p3 = others[p3_idx]

    return (np.stack(([[0., 0.], p2, p3])) + center).astype(float)


# ---------------------------------------------------------------------------
# Affine estimation
# ---------------------------------------------------------------------------

def estimate_affine(img_ac: np.ndarray, ref_ac: np.ndarray, nms_size: int = 31, n_peaks: int = 15) -> np.ndarray:
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
    img_peaks = non_maximum_suppression(img_ac, nms_size)
    img_peaks = select_lattice_peaks(img_peaks, n_peaks=n_peaks)

    ref_peaks = non_maximum_suppression(ref_ac, nms_size)
    ref_peaks = select_lattice_peaks(ref_peaks, n_peaks=n_peaks)

    img_pts = detect_interest_points(img_peaks)
    ref_pts = detect_interest_points(ref_peaks)

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
    return cv2.warpAffine(
        image.astype(np.float32), M, (W, H),
        flags=interpolation, borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    )

    # angle = abs(np.arctan2(M[1, 0], M[0, 0])) % (np.pi / 2)
    # s, c = np.sin(angle), np.cos(angle)

    # if min(H, W) <= 2 * s * c * max(H, W):
    #     crop_h = crop_w = int(min(H, W) / (2 * max(s, c) + 1e-9))
    # else:
    #     d = c * c - s * s + 1e-9
    #     crop_h = int((H * c - W * s) / d)
    #     crop_w = int((W * c - H * s) / d)

    # cy, cx = H // 2, W // 2
    # crop_h, crop_w = max(crop_h, 1), max(crop_w, 1)
    # return warped[
    #     cy - crop_h // 2 : cy + crop_h // 2,
    #     cx - crop_w // 2 : cx + crop_w // 2,
    # ].astype(image.dtype)


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
    tiles = img.reshape(H2 // up, up, W2 // up, up).transpose(1, 3, 0, 2).copy()  # (up, up, Ty, Tx)

    if tile_mode == "symmetric":
        tiles[:, :, 1::2, :] = tiles[::-1, :, 1::2, :]
        tiles[:, :, :, 1::2] = tiles[:, ::-1, :, 1::2]

    return np.sum(tiles, axis=(2, 3))


# ---------------------------------------------------------------------------
# Translation via correlation on summed block
# ---------------------------------------------------------------------------

def correct_translation_block(
    summed_block: np.ndarray,
    patch_size: int,
    key: int,
    upsample_factor: int = 2,
) -> np.ndarray:
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
    shifted_block : ndarray of shape (up, up)
    """
    ref_up = upsample_patch(generate_base_patch(patch_size, key), factor=upsample_factor).astype(np.float64)

    blocks = summed_block - summed_block.mean()
    ref = ref_up - ref_up.mean()

    list_blocks = [blocks, blocks[::-1], blocks[:, ::-1], blocks[::-1, ::-1]]

    corr = [correlation_fft(list_blocks[i], ref) for i in range(4)]
    corr_idx = np.argmax([np.max(corr[j]) for j in range(4)])

    peak_y, peak_x = np.unravel_index(np.argmax(corr[corr_idx]), corr[corr_idx].shape)
    cy, cx = corr[corr_idx].shape[0] // 2, corr[corr_idx].shape[1] // 2

    return np.roll(list_blocks[corr_idx], shift=(-peak_y + cy, -peak_x + cx), axis=(0, 1))


# def correct_translation_block(
#     summed_block: np.ndarray, shift_row: int, shift_col: int
# ) -> np.ndarray:
#     """
#     Apply a circular shift to summed_block to align it with the reference.

#     Parameters
#     ----------
#     summed_block       : ndarray of shape (up, up)
#     shift_row, shift_col : int

#     Returns
#     -------
#     aligned : ndarray of shape (up, up)
#     """
#     return np.roll(summed_block, (-shift_row, -shift_col), axis=(0, 1))


# ---------------------------------------------------------------------------
# Full synchronisation pipeline
# ---------------------------------------------------------------------------

def synchronise(
    img: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    nms_size_ac: int = 31,
    upsample_factor: int = 2,
    n_peaks: int = 10,
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
    n_peaks        : int — Number of peaks to select in autocorrelation maps

    Returns
    -------
    aligned_block : ndarray of shape (patch_size * upsample_factor,
                                      patch_size * upsample_factor), float64
    """
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 else cv2.INTER_LINEAR

    reference = build_reference_patch(
        img.shape, patch_size=patch_size, key=key,
        tile_mode=tile_mode, upsample_factor=upsample_factor,
    ).astype(float) * 2 - 1

    img_ac = correlation_fft(img, img)
    ref_ac = correlation_fft(reference, reference)

    M = estimate_affine(img_ac, ref_ac, nms_size=nms_size_ac, n_peaks=n_peaks)
    img_corrected = correct_affine(img, M, interpolation=interp)

    corrected_ac = correlation_fft(img_corrected, reference)

    M_translation = estimate_affine(corrected_ac, ref_ac, nms_size=nms_size_ac, n_peaks=n_peaks//2)

    # center_ref = get_center_peak(select_lattice_peaks(non_maximum_suppression(ref_ac, size=nms_size_ac), n_peaks=n_peaks))
    # center_corr = get_center_peak(select_lattice_peaks(non_maximum_suppression(corrected_ac, size=nms_size_ac), n_peaks=n_peaks))

    # dx, dy = center_corr - center_ref  # careful: (x, y)

    # M = np.float32([
    #     [1, 0, dx],
    #     [0, 1, dy],
    # ])

    result = correct_affine(img_corrected, M_translation, interpolation=interp)

    return result