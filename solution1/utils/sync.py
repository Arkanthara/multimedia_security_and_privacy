"""
utils/sync.py
=============
Frequency-domain synchronisation for watermark decoding.

Pipeline
--------
1.  (Optional) Centre-crop the image and reference for fast autocorrelation.
    The crop is drawn from the same centre position on both, preserving the
    tiling phase — the rotation/scale result is identical to using the full image.
2.  Compute autocorrelations via the Wiener-Khinchin theorem (rfft2).
3.  Detect lattice peaks via Non-Maximum Suppression (NMS).
4.  Extract centred interest points: the DC peak maps to (0, 0) and its
    n_neighbors closest neighbours are returned as displacement vectors.
5.  Estimate rotation + scale (no translation) from matched centred point sets.
6.  Apply rotation/scale around the image centre to the *full* image.
    The output canvas is the tight bounding box of the rotated image — no
    pixel is cropped; the canvas expands to fit.
7.  Estimate sub-tile translation and flip mode by correlating summed tiles
    against the reference patch.
8.  Apply translation to the rotated image.
    The output canvas grows by |dx| × |dy| so every pixel is kept.
9.  Return the aligned image and the detected flip mode.

Canvas sizes
------------
Both warp functions return an image *larger* than their input whenever the
transform would push pixels outside the original frame:

* ``warp_rotation_scale``  →  tight bounding box of the rotated corners
  (typically W_out × H_out > W × H for any non-zero rotation).
* ``warp_translation``     →  (W + |dx|) × (H + |dy|)

The caller (e.g. model.py) may then call ``sum_blocks`` on the returned image.

Performance notes
-----------------
* rfft2/irfft2 replace fft2/ifft2 everywhere: the one-sided spectrum has
  ~half as many elements → ~2× speedup on transforms and complex multiply.
* cv2.dilate with a rectangular kernel runs in O(N·k) — significantly faster
  than scipy.ndimage.maximum_filter for the large NMS windows produced by the
  default nms_size_ac formula.
* The reference FFT conjugate is computed once and reused for all tile
  cross-correlations.
* All four flip-variant cross-correlations are derived analytically in the
  frequency domain from a single rfft2 per tile (no extra transforms).
"""

from __future__ import annotations

import numpy as np
import cv2

from utils.patch import build_reference_patch_bipolar, upsample_patch, generate_base_patch


# ---------------------------------------------------------------------------
# Cross-correlation
# ---------------------------------------------------------------------------

def correlation_fft(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """
    Normalised cross-correlation via the Wiener-Khinchin theorem.

    ``crosscorr = irfft2(rfft2(img1) · conj(rfft2(img2)))``, shifted so that
    DC is at the array centre.

    Uses rfft2/irfft2 for ~2× speedup on real-valued inputs.

    Parameters
    ----------
    img1 : (H, W) real array
    img2 : (H, W) real array — zero-padded to img1's shape if smaller

    Returns
    -------
    crosscorr : (H, W) float64, DC at centre
    """
    def _norm(a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float64)
        return (a - a.mean()) / (a.std() + 1e-8)

    F1 = np.fft.rfft2(_norm(img1))
    F2 = np.fft.rfft2(_norm(img2), s=img1.shape)
    return np.fft.fftshift(np.fft.irfft2(F1 * np.conj(F2), s=img1.shape))


# ---------------------------------------------------------------------------
# Non-maximum suppression
# ---------------------------------------------------------------------------

def non_maximum_suppression(image: np.ndarray, size: int = 31) -> np.ndarray:
    """
    Retain only strict local maxima within a ``size × size`` neighbourhood.

    Uses cv2.dilate with a rectangular kernel — O(N·k) sliding-window
    algorithm, significantly faster than scipy.ndimage.maximum_filter for
    large kernel sizes.

    Parameters
    ----------
    image : (H, W) real array
    size  : neighbourhood side length (odd recommended)

    Returns
    -------
    nms : (H, W) — same dtype as image, zero everywhere except at local maxima
    """
    img32   = image.astype(np.float32)
    kernel  = np.ones((size, size), dtype=np.uint8)
    dilated = cv2.dilate(img32, kernel)
    return (img32 == dilated) * image


# ---------------------------------------------------------------------------
# Peak visualisation helper
# ---------------------------------------------------------------------------

def expand_peaks(image: np.ndarray, expansion_size: int = 3) -> np.ndarray:
    """
    Expand each detected peak into a small filled square for visualisation.

    Parameters
    ----------
    image          : (H, W) binary or valued peak map
    expansion_size : half-side of the square in pixels

    Returns
    -------
    expanded : (H, W), same dtype, values in {0, 1}
    """
    expanded = np.zeros_like(image)
    for y, x in np.argwhere(image > 0):
        expanded[
            max(0, y - expansion_size): min(image.shape[0], y + expansion_size + 1),
            max(0, x - expansion_size): min(image.shape[1], x + expansion_size + 1),
        ] = 1
    return expanded


# ---------------------------------------------------------------------------
# Interest-point detection
# ---------------------------------------------------------------------------

def detect_interest_points(
    peaks: np.ndarray,
    n_neighbors: int = 4,
) -> np.ndarray:
    """
    Return centred lattice interest points.

    The peak closest to the image centre is taken as the DC (origin) peak and
    placed at (0, 0). The ``n_neighbors`` nearest peaks are returned as
    displacement vectors, sorted counter-clockwise (standard mathematical
    convention with y-axis pointing up).

    All coordinates are in (row, col) order.

    Parameters
    ----------
    peaks       : (H, W) peak map — non-zero at detected peak locations
    n_neighbors : number of neighbouring peaks to return

    Returns
    -------
    points : (1 + n_neighbors, 2) float64
             Row 0 is always (0, 0) — the DC / centre peak.
             Rows 1 … n_neighbors are displacement vectors sorted by angle.

    Raises
    ------
    ValueError if fewer than n_neighbors + 1 peaks are found.
    """
    ys, xs  = np.nonzero(peaks)
    n_found = len(ys)

    if n_found < n_neighbors + 1:
        raise ValueError(
            f"Need at least {n_neighbors + 1} peaks, found {n_found}."
        )

    all_pts = np.stack([ys, xs], axis=1).astype(np.float64)

    # DC peak: closest to the image centre
    H, W   = peaks.shape
    centre = np.array([H / 2.0, W / 2.0])
    p0     = all_pts[np.argmin(np.linalg.norm(all_pts - centre, axis=1))]

    # Displacement vectors from p0, excluding p0 itself
    vecs  = all_pts - p0
    dists = np.linalg.norm(vecs, axis=1)
    valid = dists > 0
    vecs  = vecs[valid]
    dists = dists[valid]

    if len(vecs) < n_neighbors:
        raise ValueError(
            f"Need at least {n_neighbors} neighbours around the centre peak, "
            f"found {len(vecs)}."
        )

    # Keep the n_neighbors closest and sort CCW
    vecs   = vecs[np.argsort(dists)[:n_neighbors]]
    angles = np.arctan2(-vecs[:, 0], vecs[:, 1])   # negate row → standard CCW
    vecs   = vecs[np.argsort(angles)]

    return np.vstack([np.zeros((1, 2), dtype=np.float64), vecs])


# ---------------------------------------------------------------------------
# Rotation + scale estimation (no translation)
# ---------------------------------------------------------------------------

def estimate_rotation_scale(
    img_autocorr: np.ndarray,
    ref_autocorr: np.ndarray,
    nms_size: int = 31,
    n_neighbors: int = 4,
) -> np.ndarray:
    """
    Estimate the rotation + scale transform that maps the image lattice to the
    reference lattice.

    Both interest-point sets are centred at the origin (DC peak → (0, 0)), so
    the recovered transform is a pure rotation + scale with no translational
    component.  The transform maps reference-centred coordinates to
    image-centred coordinates (ref → img direction).

    Parameters
    ----------
    img_autocorr : (H, W) image autocorrelation
    ref_autocorr : (H, W) reference autocorrelation
    nms_size     : NMS neighbourhood size
    n_neighbors  : number of lattice neighbours to match

    Returns
    -------
    M : (2, 3) float64 OpenCV affine matrix
        Translation column is identically zero (pure rotation + scale).

    Raises
    ------
    ValueError if RANSAC fails to find a valid transform.
    """
    img_peaks = non_maximum_suppression(img_autocorr, nms_size)
    ref_peaks = non_maximum_suppression(ref_autocorr, nms_size)

    img_pts = detect_interest_points(img_peaks, n_neighbors)
    ref_pts = detect_interest_points(ref_peaks, n_neighbors)

    # OpenCV works in (x, y) = (col, row) order
    ref_xy = ref_pts[:, ::-1].astype(np.float32)
    img_xy = img_pts[:, ::-1].astype(np.float32)

    # Points centred at (0,0) → estimateAffine2D recovers rotation + scale only
    M, _ = cv2.estimateAffine2D(
        ref_xy,
        img_xy,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=5000,
        confidence=0.999,
        refineIters=20,
    )

    if M is None:
        raise ValueError("Rotation/scale estimation failed (RANSAC returned None).")

    M = M.astype(np.float64)
    M[:, 2] = 0.0   # enforce zero translation
    return M


# ---------------------------------------------------------------------------
# Rotation + scale warp — expanded canvas, no pixel loss
# ---------------------------------------------------------------------------

# def warp_rotation_scale(
#     image: np.ndarray,
#     M_rot_scale: np.ndarray,
#     interpolation: int = cv2.INTER_LINEAR,
# ) -> np.ndarray:
#     """
#     Apply a rotation + scale CORRECTION to the image.

#     Transform convention
#     --------------------
#     ``M_rot_scale`` must be the **correction** matrix (img → ref), i.e. the
#     matrix returned directly by ``estimate_rotation_scale``.  Do NOT invert it
#     before passing – this function uses it as the forward transform as-is.

#     The output canvas is the tight axis-aligned bounding box of the four
#     transformed source corners, so **no source pixel is ever clipped**.
#     Regions outside the warped content are filled with zeros (BORDER_CONSTANT).

#     Parameters
#     ----------
#     image        : (H, W[, C]) input image (the attacked image to correct)
#     M_rot_scale  : (2, 3) or (3, 3) affine matrix — the correction (img→ref);
#                    only the top-left 2×2 block is used (translation must be 0)
#     interpolation: OpenCV interpolation flag

#     Returns
#     -------
#     warped : ndarray — same dtype as image, shape (new_H, new_W[, C])
#     """
#     H, W   = image.shape[:2]
#     cx, cy = W / 2.0, H / 2.0

#     # A_fwd  : correction = forward transform  (img → ref, i.e. src → dst)
#     # A_bwd  : its inverse = backward transform (ref → img, i.e. dst → src)
#     #          warpAffine uses the backward / inverse-map convention
#     A_fwd = M_rot_scale[:2, :2].copy()
#     A_bwd = np.linalg.inv(A_fwd)

#     # ── 1. Tight bounding box of the warped image in destination space ───────
#     #
#     #   Forward transform centred at (cx, cy):
#     #       p_dst = A_fwd @ (p_src - c) + c
#     #
#     corners_src = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
#     corners_dst = (A_fwd @ (corners_src - [cx, cy]).T).T + [cx, cy]

#     x_min = np.floor(corners_dst[:, 0].min()).astype(int)
#     x_max = np.ceil (corners_dst[:, 0].max()).astype(int)
#     y_min = np.floor(corners_dst[:, 1].min()).astype(int)
#     y_max = np.ceil (corners_dst[:, 1].max()).astype(int)

#     new_W = x_max - x_min
#     new_H = y_max - y_min

#     # ── 2. Build the warpAffine matrix  (expanded canvas → source) ──────────
#     #
#     #   A pixel at p_new in the expanded canvas maps to
#     #   p_old = p_new + (x_min, y_min) in the original dst coordinate frame.
#     #   The backward transform then gives the source coordinate:
#     #
#     #       p_src = A_bwd @ (p_old  - c) + c
#     #             = A_bwd @ (p_new + [x_min, y_min] - c) + c
#     #
#     def _T(tx: float, ty: float) -> np.ndarray:
#         return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]], dtype=np.float64)

#     A_bwd_3x3          = np.eye(3, dtype=np.float64)
#     A_bwd_3x3[:2, :2]  = A_bwd

#     M_warp = (_T(cx, cy) @ A_bwd_3x3 @ _T(-cx, -cy) @ _T(x_min, y_min))[:2]

#     return cv2.warpAffine(
#         image,
#         M_warp,
#         (new_W, new_H),
#         flags=interpolation,
#         borderMode=cv2.BORDER_CONSTANT,
#         borderValue=0,
#     )

def warp_rotation_scale(
    image: np.ndarray,
    M_rot_scale: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Apply a rotation + scale transform centred on the image.

    The output canvas is the tight axis-aligned bounding box of the four
    rotated image corners, so **no pixel is ever cropped**.  For any non-zero
    rotation the output is larger than the input; the area outside the original
    image is filled with zeros (BORDER_CONSTANT).

    Transform convention
    --------------------
    ``M_rot_scale`` (as returned by ``estimate_rotation_scale``) maps
    reference-centred coordinates to image-centred coordinates (ref → img).
    This function inverts it so the image is mapped *into* the reference
    frame (img → ref), and applies the resulting warp centred on the image.

    Parameters
    ----------
    image        : (H, W[, C]) input image
    M_rot_scale  : (2, 3) affine matrix — translation column must be zero
    interpolation: OpenCV interpolation flag

    Returns
    -------
    warped : ndarray — same dtype as image, shape (new_H, new_W[, C])
             new_H and new_W are >= H and W respectively.
    """
    H, W = image.shape[:2]
    cx, cy = (W - 1) / 2.0, (H - 1) / 2.0

    # ref -> img
    A = M_rot_scale[:2, :2]

    # correction = img -> ref
    A_inv = np.linalg.inv(A)

    # Build affine transform centered on image
    M = np.zeros((2, 3), dtype=np.float64)
    M[:2, :2] = A_inv

    # translation so rotation/scale happens around center
    M[:, 2] = [cx, cy] - A_inv @ np.array([cx, cy])

    corrected = cv2.warpAffine(
        image,
        M,
        (W, H),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )

    return corrected


# ---------------------------------------------------------------------------
# Translation warp — expanded canvas, no pixel loss
# ---------------------------------------------------------------------------

def warp_translation(
    image: np.ndarray,
    shift: np.ndarray,
    interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """
    Apply a (dy, dx) translation to the image.

    The output canvas grows by ``|dx|`` columns and ``|dy|`` rows so that
    **every source pixel appears in the output** — nothing is cropped.
    The empty border (pixels that have no source) is filled with zeros.

    Canvas expansion rules
    ----------------------
    shift = (dy, dx) means image content moves by (+dy, +dx) in the output.

    * dx > 0  →  content shifts right;  ``dx`` zero columns added on the left.
    * dx < 0  →  content shifts left;   ``|dx|`` zero columns added on the right.
    * dy > 0  →  content shifts down;   ``dy`` zero rows added on the top.
    * dy < 0  →  content shifts up;     ``|dy|`` zero rows added on the bottom.

    Parameters
    ----------
    image  : (H, W[, C]) input image
    shift  : (dy, dx) pixel shift
    interpolation: OpenCV interpolation flag

    Returns
    -------
    shifted : ndarray — same dtype as image, shape (H + |dy|, W + |dx|[, C])
    """
    H, W   = image.shape[:2]
    dy, dx = float(shift[0]), float(shift[1])

    # warpAffine dst→src convention: src = (x + tx, y + ty)
    # Forward: src(xs, ys) appears at dst(xs - tx, ys - ty) = dst(xs + dx, ys + dy)
    tx = -dx
    ty = -dy

    # Canvas offset to shift all dst coordinates into the non-negative quadrant
    x_offset = max(0.0,  tx)   # = max(0, -dx)
    y_offset = max(0.0,  ty)   # = max(0, -dy)

    new_W = W + abs(int(round(dx)))
    new_H = H + abs(int(round(dy)))

    # In the expanded canvas:
    #   src_x = (x_new - x_offset) + tx = x_new + (tx - x_offset) = x_new + min(tx, 0)
    #   src_y = (y_new - y_offset) + ty = y_new + (ty - y_offset) = y_new + min(ty, 0)
    M = np.array(
        [[1.0, 0.0, min(tx, 0.0)],
         [0.0, 1.0, min(ty, 0.0)]],
        dtype=np.float64,
    )

    return cv2.warpAffine(
        image,
        M,
        (new_W, new_H),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


# ---------------------------------------------------------------------------
# Translation + flip estimation
# ---------------------------------------------------------------------------

def estimate_translation_and_flip(
    image: np.ndarray,
    key: int,
    patch_size: int,
    upsample_factor: int,
    max_tiles: int = 16,
) -> tuple[np.ndarray, str]:
    """
    Estimate sub-tile translation and flip mode by correlating summed tiles
    against the reference patch.

    All four flip variants (normal, flip_y, flip_x, flip_xy) are evaluated
    analytically in the frequency domain from a single rfft2 per tile, so
    no extra transforms are needed.

    Parameters
    ----------
    image           : (H, W) float array — after rotation correction
    key             : watermark key
    patch_size      : logical patch side length
    upsample_factor : must match the value used at encode time
    max_tiles       : maximum number of tiles to average over

    Returns
    -------
    shift     : (dy, dx) int array — pixel shift to apply
    flip_mode : one of {"normal", "flip_y", "flip_x", "flip_xy"}
    """
    up   = patch_size * upsample_factor
    H, W = image.shape

    # Build and normalise the upsampled reference patch (float32 for speed)
    ref = upsample_patch(generate_base_patch(patch_size, key), upsample_factor)
    ref = ref.astype(np.float32)
    ref = (ref - ref.mean()) / (ref.std() + 1e-8)

    # Pad image to an exact multiple of the tile size
    img = np.pad(
        image.astype(np.float32),
        ((0, (-H) % up), (0, (-W) % up)),
    )
    H2, W2 = img.shape
    Ty, Tx = H2 // up, W2 // up

    tiles = img.reshape(Ty, up, Tx, up).transpose(0, 2, 1, 3)

    # Select up to max_tiles tiles on a coarse grid, closest to the image centre
    ey, ex  = np.arange(0, Ty, 2), np.arange(0, Tx, 2)
    yy, xx  = np.meshgrid(ey, ex, indexing="ij")
    coords  = np.stack([yy.ravel(), xx.ravel()], axis=1)
    cy, cx  = Ty / 2.0, Tx / 2.0
    dist    = (coords[:, 0] - cy) ** 2 + (coords[:, 1] - cx) ** 2
    coords  = coords[np.argsort(dist)[:max_tiles]]

    selected = tiles[coords[:, 0], coords[:, 1]]               # (N, up, up)

    # Normalise once — flip operations are mean/std invariant
    mu       = selected.mean(axis=(-2, -1), keepdims=True)
    sigma    = selected.std( axis=(-2, -1), keepdims=True) + 1e-8
    selected = (selected - mu) / sigma

    # One rfft2 per tile; derive the three other flip variants analytically:
    #
    #   flip_y  a[::-1, :]    →  F[(N-k) % N, l]        (reverse k-axis)
    #   flip_xy a[::-1, ::-1] →  conj(F)                 (Hermitian symmetry)
    #   flip_x  a[:, ::-1]    →  conj(F[(N-k) % N, l])  (= conj of flip_y)
    F_tiles    = np.fft.rfft2(selected, axes=(-2, -1))          # (N, up, up//2+1)
    F_ref_conj = np.conj(np.fft.rfft2(ref))                     # (up, up//2+1)

    F_flip_y  = np.concatenate(
        [F_tiles[:, :1, :], F_tiles[:, 1:, :][:, ::-1, :]], axis=1
    )
    F_flip_xy = np.conj(F_tiles)
    F_flip_x  = np.conj(F_flip_y)

    # (N, 4, up, up//2+1)
    F_variants = np.stack([F_tiles, F_flip_y, F_flip_x, F_flip_xy], axis=1)

    corr = np.fft.irfft2(F_variants * F_ref_conj, s=(up, up), axes=(-2, -1))
    corr = np.fft.fftshift(corr, axes=(-2, -1))                 # (N, 4, up, up)

    N    = len(coords)
    flat = corr.reshape(N, 4, -1)
    idx  = np.argmax(flat, axis=-1)                             # (N, 4)
    scores    = flat[np.arange(N)[:, None], np.arange(4), idx]
    best_mode = np.argmax(scores, axis=1)                       # (N,)

    dy = (idx // up)[np.arange(N), best_mode] - up // 2
    dx = (idx %  up)[np.arange(N), best_mode] - up // 2

    shift  = np.round(
        np.median(np.stack([dy, dx], axis=1), axis=0)
    ).astype(int)

    counts = np.bincount(best_mode, minlength=4)
    modes  = ["normal", "flip_y", "flip_x", "flip_xy"]
    return shift, modes[int(np.argmax(counts))]


# ---------------------------------------------------------------------------
# Block summation
# ---------------------------------------------------------------------------

def sum_blocks(
    image: np.ndarray,
    patch_size: int,
    tile_mode: str = "normal",
    upsample_factor: int = 2,
    flip_mode: str = "normal",
) -> np.ndarray:
    """
    Fold all non-overlapping (up × up) tiles into a single block by summation.

    When ``tile_mode == "symmetric"``, every other tile is mirror-flipped
    before summation to undo the symmetric tiling applied during encoding.

    Parameters
    ----------
    image           : (H, W) array
    patch_size      : logical patch side length (before upsampling)
    tile_mode       : {"normal", "symmetric"} — must match encode-time setting
    upsample_factor : must match encode-time value
    flip_mode       : {"normal", "flip_y", "flip_x", "flip_xy"} — flip
                      correction detected by ``estimate_translation_and_flip``

    Returns
    -------
    block : (up, up) float64  where  up = patch_size × upsample_factor
    """
    up   = patch_size * upsample_factor
    H, W = image.shape

    img = np.pad(
        image.astype(np.float64),
        ((0, (-H) % up), (0, (-W) % up)),
    )
    H2, W2 = img.shape

    # tiles shape: (up, up, Ty, Tx)
    tiles = (
        img.reshape(H2 // up, up, W2 // up, up)
           .transpose(1, 3, 0, 2)
           .copy()
    )

    if tile_mode == "symmetric":
        tiles[:, :, 1::2, :] = tiles[::-1, :, 1::2, :]
        tiles[:, :, :, 1::2] = tiles[:, ::-1, :, 1::2]

    block = tiles.sum(axis=(2, 3))

    if flip_mode == "flip_y":
        block = block[::-1, :]
    elif flip_mode == "flip_x":
        block = block[:, ::-1]
    elif flip_mode == "flip_xy":
        block = block[::-1, ::-1]

    return block


# ---------------------------------------------------------------------------
# Main synchronisation pipeline
# ---------------------------------------------------------------------------

def synchronise(
    img: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    nms_size_ac: int | None = None,
    upsample_factor: int = 2,
    reference: np.ndarray | None = None,
    crop_size: int | None = None,
    n_neighbors: int = 4,
    max_tiles: int = 16,
) -> tuple[np.ndarray, str]:
    """
    Align ``img`` to the reference watermark grid.

    The alignment is performed in two sequential stages, each producing a
    *larger* output canvas so that no image content is ever cropped:

    1. **Rotation + scale** — estimated from autocorrelation lattice peaks
       (optionally on a centre crop for speed), then applied to the full image.
       The output canvas is the tight bounding box of the rotated corners.

    2. **Translation** — estimated by correlating summed tiles of the rotated
       image against the reference patch, then applied.  The canvas grows by
       ``|dx|`` × ``|dy|`` pixels to accommodate the shift.

    Because the two stages change the image size, they are applied as two
    separate warpAffine calls rather than being composed.

    Parameters
    ----------
    img             : (H, W) input image, uint8 or float
    patch_size      : logical patch side length
    key             : watermark key
    tile_mode       : {"normal", "symmetric"} — must match encode-time setting
    nms_size_ac     : NMS window for autocorrelation peak detection.
                      Default: ``patch_size * upsample_factor * 2 - 1``
                      (spans one full watermark period).
    upsample_factor : must match the value used at encode time
    reference       : optional pre-built full-size reference array.  When
                      supplied the function skips ``build_reference_patch_bipolar``
                      (saves one full-image FFT pair).
    crop_size       : side length (pixels) of the centre crop used for the
                      autocorrelation step only.  Must be a multiple of
                      ``patch_size * upsample_factor * 2``.
                      Pass ``None`` (default) to use the full image for maximum
                      robustness.  The crop never affects which pixels appear
                      in the output — only estimation speed changes.
    n_neighbors     : number of lattice neighbours used for rotation estimation
    max_tiles       : maximum number of tiles averaged for translation estimation

    Returns
    -------
    img_aligned : ndarray, same dtype as input
                  Full image after rotation + translation correction.
                  Shape is (H_out, W_out) where H_out >= H and W_out >= W.
    flip_mode   : str — one of {"normal", "flip_y", "flip_x", "flip_xy"}
    """
    H, W   = img.shape[:2]
    up     = patch_size * upsample_factor
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 else cv2.INTER_LINEAR

    if nms_size_ac is None:
        nms_size_ac = up * 2 - 1

    # ------------------------------------------------------------------
    # 1. Build reference at full image size.
    #    Must always be full-size so that the tiling phase at the centre
    #    of the image is correct.
    # ------------------------------------------------------------------
    if reference is None:
        ref_full = build_reference_patch_bipolar(
            (H, W),
            patch_size=patch_size,
            key=key,
            tile_mode=tile_mode,
            upsample_factor=upsample_factor,
        ).astype(np.float64)
    else:
        ref_full = np.asarray(reference, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. Optional centre crop — affects autocorrelation speed only.
    #    Both image and reference are cropped from the same centre
    #    position so their relative tiling phase is preserved.
    # ------------------------------------------------------------------
    if crop_size is not None:
        half = crop_size // 2
        cy_, cx_ = H // 2, W // 2
        y0 = max(0, cy_ - half);  y1 = min(H, cy_ + half)
        x0 = max(0, cx_ - half);  x1 = min(W, cx_ + half)
        img_crop = img[y0:y1, x0:x1]
        ref_crop = ref_full[y0:y1, x0:x1]
    else:
        img_crop = img
        ref_crop = ref_full

    # ------------------------------------------------------------------
    # 3. Autocorrelations (rfft2 — one-sided spectrum for speed)
    # ------------------------------------------------------------------

    img_autocorr = correlation_fft(img_crop, img_crop)
    ref_autocorr = correlation_fft(ref_crop, ref_crop)

    # ------------------------------------------------------------------
    # 4. Rotation + scale estimation from centred interest points.
    #    M_rot maps reference-centred → image-centred (ref → img).
    # ------------------------------------------------------------------
    M_rot = estimate_rotation_scale(
        img_autocorr,
        ref_autocorr,
        nms_size=nms_size_ac,
        n_neighbors=n_neighbors,
    )

    # ------------------------------------------------------------------
    # 5. Rotation + scale warp on the full original image.
    #    The canvas expands to the tight bounding box of the rotated
    #    corners — every source pixel is preserved.
    # ------------------------------------------------------------------
    img_rotated = warp_rotation_scale(img, M_rot, interpolation=interp)

    # ------------------------------------------------------------------
    # 6. Translation + flip estimation on the full rotated image.
    # ------------------------------------------------------------------
    shift, flip_mode = estimate_translation_and_flip(
        img_rotated.astype(np.float64),
        key,
        patch_size,
        upsample_factor,
        max_tiles=max_tiles,
    )

    # ------------------------------------------------------------------
    # 7. Translation warp on the full rotated image.
    #    The canvas grows by |dx| × |dy| — every pixel is preserved.
    # ------------------------------------------------------------------
    img_aligned = warp_translation(img_rotated, shift, interpolation=interp)

    return img_aligned, flip_mode


def synchronise_debug(
    img: np.ndarray,
    patch_size: int = 32,
    key: int = 0,
    tile_mode: str = "symmetric",
    nms_size_ac: int | None = None,
    upsample_factor: int = 2,
    reference: np.ndarray | None = None,   # pre-built watermark reference (or None → built internally)
    crop_size: int | None = None,
    n_neighbors: int = 4,
    max_tiles: int = 16,
    original: np.ndarray | None = None,    # original clean image — used only for visual overlays
) -> tuple[np.ndarray, str]:
    """
    Debug version of synchronise — plots every step inline (Jupyter).
    Signature mirrors synchronise() exactly, plus an optional `original`
    parameter for visual overlays (the clean image before attacks).
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.colors as mcolors

    H, W   = img.shape[:2]
    up     = patch_size * upsample_factor
    interp = cv2.INTER_NEAREST if img.dtype == np.uint8 else cv2.INTER_LINEAR

    if nms_size_ac is None:
        nms_size_ac = up * 2 - 1

    # ── helpers ──────────────────────────────────────────────────────

    def _to_gray(a):
        """Ensure 2-D float64."""
        a = np.asarray(a, dtype=np.float64)
        if a.ndim == 3:
            a = a.mean(axis=-1)
        return a

    def _fit_to(src, th, tw):
        """Centre-crop OR zero-pad src (2-D) to exactly (th, tw)."""
        src = _to_gray(src)
        sh, sw = src.shape

        # ── crop if src is larger ──────────────────────────────────────
        if sh > th:
            y0 = (sh - th) // 2
            src = src[y0:y0 + th, :]
            sh  = th
        if sw > tw:
            x0 = (sw - tw) // 2
            src = src[:, x0:x0 + tw]
            sw  = tw

        # ── pad if src is smaller ──────────────────────────────────────
        out = np.zeros((th, tw), dtype=np.float64)
        y0, x0 = (th - sh) // 2, (tw - sw) // 2
        out[y0:y0 + sh, x0:x0 + sw] = src
        return out

    def _overlay_original(transformed):
        """Overlay original (red) behind transformed (grey), matching canvas size."""
        ref_img     = original if original is not None else img
        th, tw      = transformed.shape[:2]
        orig_fitted = _fit_to(ref_img, th, tw)
        plt.imshow(original)
        plt.imshow(transformed, cmap="seismic", norm=mcolors.CenteredNorm(), alpha=0.5)

    # ════════════════════════════════════════════════════════════════
    # STEP 0 — Input image (+ crop box if crop_size is set)
    # ════════════════════════════════════════════════════════════════
    plt.figure(figsize=(5, 5))
    plt.imshow(_to_gray(img), cmap="gray")
    if crop_size is not None:
        cy, cx = H // 2, W // 2
        half   = crop_size // 2
        y0c, y1c = max(0, cy - half), min(H, cy + half)
        x0c, x1c = max(0, cx - half), min(W, cx + half)
        plt.gca().add_patch(mpatches.Rectangle(
            (x0c, y0c), x1c - x0c, y1c - y0c,
            lw=2, edgecolor="red", facecolor="none",
            label=f"autocorr crop [{y1c-y0c}×{x1c-x0c}]",
        ))
        plt.legend(fontsize=7)
    plt.title(f"Step 0 — Input [{H}×{W}]  patch={patch_size}  up={upsample_factor}  key={key}", fontsize=8)
    plt.axis("off"); plt.tight_layout(); plt.show()

    # ════════════════════════════════════════════════════════════════
    # STEP 1 — Build watermark reference + (optional) centre crop
    # ════════════════════════════════════════════════════════════════
    if reference is None:
        ref_full = build_reference_patch_bipolar(
            (H, W), patch_size=patch_size, key=key,
            tile_mode=tile_mode, upsample_factor=upsample_factor,
        ).astype(np.float64)
    else:
        ref_full = _to_gray(reference)   # ensure 2-D float64

    if crop_size is not None:
        cy, cx = H // 2, W // 2
        half   = crop_size // 2
        y0c, y1c = max(0, cy - half), min(H, cy + half)
        x0c, x1c = max(0, cx - half), min(W, cx + half)
        img_crop = _to_gray(img)[y0c:y1c, x0c:x1c]
        ref_crop = ref_full[y0c:y1c, x0c:x1c]
    else:
        img_crop = _to_gray(img)
        ref_crop = ref_full

    ch, cw = img_crop.shape
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(img_crop, cmap="RdBu_r")
    plt.title(f"Image (residual) crop [{ch}×{cw}]", fontsize=8); plt.colorbar(fraction=.03); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(ref_crop,  cmap="RdBu_r")
    plt.title("Watermark reference crop (bipolar)", fontsize=8); plt.colorbar(fraction=.03); plt.axis("off")
    plt.suptitle("Step 1 — Crops used for autocorrelation", fontsize=9)
    plt.tight_layout(); plt.show()

    # ════════════════════════════════════════════════════════════════
    # STEP 2 — Autocorrelations (cross-correlation of each signal with itself)
    # ════════════════════════════════════════════════════════════════
    img_ac = correlation_fft(img_crop, img_crop)
    ref_ac = correlation_fft(ref_crop, ref_crop)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1); plt.imshow(img_ac, cmap="inferno")
    plt.title("Image autocorrelation",     fontsize=8); plt.axis("off")
    plt.subplot(1, 2, 2); plt.imshow(ref_ac, cmap="inferno")
    plt.title("Reference autocorrelation", fontsize=8); plt.axis("off")
    plt.suptitle("Step 2 — Autocorrelations  (DC centred)", fontsize=9)
    plt.tight_layout(); plt.show()

    # ════════════════════════════════════════════════════════════════
    # STEP 3 — NMS peaks overlaid on autocorrelations
    # ════════════════════════════════════════════════════════════════
    img_nms = non_maximum_suppression(img_ac, nms_size_ac)
    ref_nms = non_maximum_suppression(ref_ac, nms_size_ac)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.imshow(img_ac, cmap="inferno")
    iy, ix = np.nonzero(img_nms)
    plt.scatter(ix, iy, c="cyan", s=15, label=f"{len(iy)} peaks")
    plt.title(f"Image NMS peaks  (nms_size={nms_size_ac})", fontsize=8); plt.legend(fontsize=7); plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(ref_ac, cmap="inferno")
    ry, rx = np.nonzero(ref_nms)
    plt.scatter(rx, ry, c="cyan", s=15, label=f"{len(ry)} peaks")
    plt.title("Reference NMS peaks", fontsize=8); plt.legend(fontsize=7); plt.axis("off")
    plt.suptitle("Step 3 — Non-maximum suppression", fontsize=9)
    plt.tight_layout(); plt.show()

    # ════════════════════════════════════════════════════════════════
    # STEP 4 — Lattice interest points & vectors
    # ════════════════════════════════════════════════════════════════
    try:
        img_pts = detect_interest_points(img_nms, n_neighbors)   # centred at (0,0)
        ref_pts = detect_interest_points(ref_nms, n_neighbors)
        ctr_img = np.array([img_ac.shape[0] / 2, img_ac.shape[1] / 2])
        ctr_ref = np.array([ref_ac.shape[0] / 2, ref_ac.shape[1] / 2])
        colors  = ["#e74c3c", "#3498db", "#2ecc71", "#f1c40f"]

        plt.figure(figsize=(10, 4))
        for i, (ac, pts, ctr, label) in enumerate([
            (img_ac, img_pts, ctr_img, "Image"),
            (ref_ac, ref_pts, ctr_ref, "Reference"),
        ]):
            plt.subplot(1, 2, i + 1)
            plt.imshow(ac, cmap="inferno")
            p0 = pts[0] + ctr   # DC peak in pixel coords
            plt.scatter(p0[1], p0[0], c="white", marker="*", s=250, zorder=10, label="DC")
            for j, (pt, col) in enumerate(zip(pts[1:], colors)):
                p = pt + ctr
                dist  = np.linalg.norm(pt)
                angle = np.degrees(np.arctan2(-pt[0], pt[1]))
                plt.scatter(p[1], p[0], c=col, s=80, zorder=6,
                            label=f"p{j+1}  {dist:.1f}px  {angle:.1f}°")
                plt.annotate("", xy=(p[1], p[0]), xytext=(p0[1], p0[0]),
                             arrowprops=dict(arrowstyle="->", color=col, lw=1.5))
            plt.title(f"{label} lattice vectors", fontsize=8)
            plt.legend(fontsize=6, loc="upper right"); plt.axis("off")
        plt.suptitle("Step 4 — Interest points  (DC → nearest neighbours)", fontsize=9)
        plt.tight_layout(); plt.show()
    except Exception as e:
        print(f"[step 4 failed] {e}")

    # ════════════════════════════════════════════════════════════════
    # STEP 5 — Rotation + scale estimation → warp
    # ════════════════════════════════════════════════════════════════
    M_rot = estimate_rotation_scale(img_ac, ref_ac, nms_size=nms_size_ac, n_neighbors=n_neighbors)

    # M_rot is ref→img (the detected attack).  The correction is its inverse.
    A2     = M_rot[:2, :2]
    A2_inv = np.linalg.inv(A2)
    rot_detected   = float(np.degrees(np.arctan2( A2[1, 0],     A2[0, 0])))
    scale_detected = float(np.sqrt(A2[0, 0]**2 + A2[1, 0]**2))
    rot_corr       = float(np.degrees(np.arctan2( A2_inv[1, 0], A2_inv[0, 0])))
    scale_corr     = float(np.sqrt(A2_inv[0, 0]**2 + A2_inv[1, 0]**2))

    print(f"[step 5] Detected attack : rotation={rot_detected:+.2f}°  scale={scale_detected:.4f}")
    print(f"[step 5] Correction applied: rotation={rot_corr:+.2f}°  scale={scale_corr:.4f}")

    img_rotated = warp_rotation_scale(img, M_rot, interpolation=interp)
    Hr, Wr = img_rotated.shape[:2]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.imshow(img, cmap="seismic", norm=mcolors.CenteredNorm())
    plt.title(f"Original [{H}×{W}]", fontsize=8); plt.axis("off")
    plt.subplot(1, 2, 2); _overlay_original(img_rotated)
    plt.title(f"Corrected [{Hr}×{Wr}]\n(red = original, centred)", fontsize=8); plt.axis("off")
    plt.suptitle(
        f"Step 5 — Rotation + scale\n"
        f"Detected attack: rot={rot_detected:+.2f}°  scale={scale_detected:.4f}   |   "
        f"Correction: rot={rot_corr:+.2f}°  scale={scale_corr:.4f}",
        fontsize=8,
    )
    plt.tight_layout(); plt.show()

    # keep using rot_corr / scale_corr labels in subsequent steps
    rot, scale = rot_corr, scale_corr

    # ════════════════════════════════════════════════════════════════
    # STEP 6 — Translation + flip estimation → warp
    # ════════════════════════════════════════════════════════════════
    shift, flip_mode = estimate_translation_and_flip(
        img_rotated.astype(np.float64), key, patch_size, upsample_factor, max_tiles=max_tiles,
    )
    dy, dx = float(shift[0]), float(shift[1])
    print(f"[step 6] translation dy={dy:+.1f}px  dx={dx:+.1f}px  flip='{flip_mode}'")

    img_aligned = warp_translation(img_rotated, shift, interpolation=interp)
    Ha, Wa = img_aligned.shape[:2]

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1); plt.imshow(img_rotated, cmap="seismic", norm=mcolors.CenteredNorm())
    plt.title(f"After rotation [{Hr}×{Wr}]", fontsize=8); plt.axis("off")
    plt.subplot(1, 2, 2); _overlay_original(img_aligned)
    plt.title(f"Translated [{Ha}×{Wa}]\n(red = original, centred)", fontsize=8); plt.axis("off")
    plt.suptitle(f"Step 6 — Translation  dy={dy:+.1f}  dx={dx:+.1f}  flip='{flip_mode}'", fontsize=9)
    plt.tight_layout(); plt.show()

    # ════════════════════════════════════════════════════════════════
    # STEP 7 — Final result vs original
    # ════════════════════════════════════════════════════════════════
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    ref_img = original if original is not None else img
    plt.imshow(ref_img)
    plt.title(f"Original (clean) [{ref_img.shape[0]}×{ref_img.shape[1]}]", fontsize=8); plt.axis("off")
    plt.subplot(1, 2, 2); _overlay_original(img_aligned)
    plt.title("Overlay  (grey=aligned, red=original)", fontsize=8); plt.axis("off")
    plt.suptitle(
        f"Step 7 — Final result\n"
        f"rot={rot:.2f}°  scale={scale:.4f}  dy={dy:+.1f}  dx={dx:+.1f}  flip='{flip_mode}'",
        fontsize=9,
    )
    plt.tight_layout(); plt.show()

    return img_aligned, flip_mode