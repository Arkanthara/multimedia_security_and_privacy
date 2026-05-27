# Solution for JPEG Trust Watermarking Benchmark

### Team `unige-msp`
### University of Geneva

Production-ready spatial-domain watermarking and robustness benchmarking toolkit.

This repository implements a full watermark encode/decode stack with:

- NVF-weighted embedding
- ECC-aware message preprocessing/postprocessing
- Frequency-domain geometric synchronization
- Benchmarking utilities in a Jupyter notebook

The main implementation is in `spatial/model.py`, with synchronization and patch logic in `utils/sync.py` and `utils/patch.py`.

## 1) What This Project Does

Given an input image and a binary message:

1. Encodes a robust watermark into the image.
2. Applies distortions/attacks (rotation, scale, crop, blur, JPEG, etc.).
3. Decodes the message from attacked images.
4. Computes BER and quality/timing metrics.

This is designed for experimentation, comparative robustness testing, and reproducible benchmarking.

## 2) Repository Structure

```text
benchmark.ipynb           # End-to-end benchmark and diagnostics
spatial/model.py          # WatermarkModel (encode/decode entry points)
utils/sync.py             # Geometric synchronization pipeline
utils/patch.py            # Keyed patch generation and bit extraction
utils/error_correction.py # Repetition + LDPC-based ECC helpers
```

## 3) Requirements

- Python 3.11+
- Core dependencies (from `pyproject.toml`):
  - numpy
  - opencv-python
  - scikit-image
  - scipy
  - pandas
  - matplotlib
  - ldpc

## 4) Quick Start

```python
import numpy as np
from skimage import data
from skimage.util import img_as_ubyte

from spatial.model import WatermarkModel

# 1) Build model
model = WatermarkModel(
    message_length=100,
    patch_size=32,
    upsample_factor=4,
    use_ecc=True,
    ecc_repetitions=5,
    msg_repetitions=1,
    tile_mode="symmetric",
    key=7,
)

# 2) Image + message
image = img_as_ubyte(data.astronaut())
watermark = np.random.default_rng(42).integers(0, 2, model.message_length, dtype=np.uint8)

# 3) Encode
watermarked = model.encode(image, watermark)

# 4) Decode
decoded = model.decode(watermarked)

ber = float(np.mean(watermark != decoded))
print(f"BER: {ber:.4f}")
```

## 5) Global Watermark Pipeline

## 5.1 Encode Pipeline

`WatermarkModel.encode(image, watermark)`:

1. Convert image to float representation.
2. Preprocess message:
   - optional repetition encoding (LDPC-based helper)
   - optional message tiling via `msg_repetitions`
3. Build keyed bipolar watermark patch over the full image:
   - base patch generation
   - bit-position injection
   - upsampling (`upsample_factor`)
   - tiling (`tile_mode`)
4. Compute NVF on embedding channel.
5. Compute per-pixel embedding strength:
   - `alpha_1 * NVF + alpha_2 * (1 - NVF)`
6. Additive embedding in selected channel.
7. Clip to valid range and return uint8 image.

## 5.2 Decode Pipeline

`WatermarkModel.decode(image)`:

1. Convert image and extract embedding channel.
2. Build full-image bipolar reference patch.
3. Compute NVF + embedding strength and estimate watermark energy map.
4. Wiener denoise channel (`scipy.signal.wiener`).
5. Compute residual: `residual = channel - denoised`.
6. Fallback extraction directly from residual (no sync), with confidence score.
7. Run geometric synchronization on residual (if possible):
   - estimate affine + translation/flip corrections
   - align to reference grid
8. Extract bits from synchronized residual and compute confidence.
9. Choose highest-confidence candidate.
10. Postprocess bits:
    - majority vote across repeated messages (`msg_repetitions`)
    - optional repetition decode (LDPC helper)
11. Return recovered message of length `message_length`.

## 6) Global Synchronization Pipeline

Implemented in `utils/sync.py` via `synchronise(...)`.

### Inputs

- `img`: residual-like 2D signal (`H x W`)
- `patch_size`: logical patch side length
- `key`: deterministic seed for reference patch
- `tile_mode`: expected tiling mode
- `nms_size_ac`: non-maximum suppression window for autocorrelation peaks
- `upsample_factor`: logical-to-pixel expansion factor
- `reference`: optional prebuilt full-image bipolar reference

### Steps

1. Compute a centered crop for fast estimation (multiple watermark periods).
2. Build (or receive) full reference, then crop it with identical crop window.
3. Compute reference autocorrelation in frequency domain.
4. Compute image autocorrelation in frequency domain.
5. Detect stable lattice peaks and estimate affine transform.
6. Affine-correct image.
7. Estimate translation and flip mode from tiled correlations.
8. Fuse affine + translation corrections.
9. Return aligned residual and inferred flip mode.

### Output

```python
aligned_residual, flip_mode = synchronise(...)
```

- `aligned_residual`: corrected 2D residual-like signal
- `flip_mode`: one of `normal`, `flip_y`, `flip_x`, `flip_xy`

## 7) Parameter Reference

## 7.1 WatermarkModel Parameters

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `alpha_1` | float | `4` | Embedding base strength component (scaled by 1/255 internally). |
| `alpha_2` | float | `20` | Complementary embedding strength component (scaled by 1/255). |
| `D` | float | `50.0` | NVF sharpness; affects strength map response to local variance. |
| `use_nvf` | bool | `True` | If `False`, embedding strength is uniform (`alpha_1`). |
| `nvf_window_size` | int | `7` | Local statistics window for NVF. |
| `message_length` | int | `100` | Final output message length (bits). |
| `use_ecc` | bool | `True` | Enables repetition-code ECC helpers. |
| `ecc_repetitions` | int | `5` | Number of repetition-code copies per bit before embedding. |
| `msg_repetitions` | int | `1` | Number of times encoded message is tiled before embedding. |
| `patch_size` | int | `32` | Logical patch width/height. |
| `upsample_factor` | int | `4` | Pixels per logical patch cell per axis. |
| `key` | int | `7` | Seed controlling patch and bit positions. |
| `tile_mode` | str | `"symmetric"` | Patch tiling convention (`normal` or `symmetric`). |
| `nms_size_ac` | int or None | auto | NMS size in synchronization autocorrelation stage. |
| `psnr_threshold` | float | `30.0` | Metadata/config threshold for quality checks. |
| `max_encode_time` | float | `5.0` | Metadata/config target encode budget (seconds). |
| `max_decode_time` | float | `1.0` | Metadata/config target decode budget (seconds). |

Important consistency constraints:

- `patch_size`, `upsample_factor`, `key`, and `tile_mode` must match between encode and decode.
- Changing ECC settings changes embedded bit count and expected decoding behavior.

## 7.2 Synchronization Parameters

| Parameter | Type | Default | Meaning |
|---|---:|---:|---|
| `img` | ndarray | required | Residual-like 2D signal. |
| `patch_size` | int | `32` | Logical patch size for periodicity estimation. |
| `key` | int | `0` | Reference seed. |
| `tile_mode` | str | `"symmetric"` | Tiling model assumed by the synchronizer. |
| `nms_size_ac` | int | `31` | NMS kernel size for autocorrelation peaks. |
| `upsample_factor` | int | `2` | Logical-to-pixel expansion factor. |
| `reference` | ndarray or None | `None` | Optional precomputed full reference to avoid recomputation. |

## 7.3 Patch/Extraction Parameters

From `utils/patch.py`:

- `generate_base_patch(patch_size, key)`: deterministic binary patch.
- `get_bit_positions(patch_size, n_bits, key)`: deterministic position sampling.
- `build_patch_image_bipolar(...)`: full-image bipolar watermark field.
- `build_reference_patch_bipolar(...)`: full-image bipolar reference field.
- `extract_bits_from_spatial(residual, patch_size, n_bits, key, tile_mode, upsample_factor, mode)`:
  - aggregates tiles,
  - samples keyed bit positions,
  - returns `(bits, confidence)`.

## 8) Running the Notebook Benchmark

Open and run `benchmark.ipynb` top-to-bottom.

Expected sections:

1. Setup and imports
2. Image loading
3. Helper functions
4. Attack definitions
5. Model + benchmark execution
6. Summary visualizations (BER/timing)
7. Per-attack diagnostics

Outputs include:

- BER per image/attack
- Encode/decode runtime
- PSNR (embedded and attacked)
- Diagnostic visualizations for residual/synchronization behavior

## 9) How Confidence-Based Decoding Works

During decode, multiple candidate bitstreams may exist:

- direct residual extraction (fallback)
- synchronized extraction

Each candidate gets a confidence score from extraction statistics. The decoder keeps the highest-confidence candidate before ECC/message postprocessing.

This improves robustness when synchronization is ambiguous or fails under severe attacks.

## 10) Performance Notes

The implementation includes practical optimizations:

- frequency-domain correlations with real FFT paths
- OpenCV dilation for fast NMS
- reused reference arrays across decode/sync steps
- crop-based geometric estimation to reduce FFT workload

Tradeoffs:

- larger `patch_size` and `upsample_factor` can improve robustness but increase compute
- stronger embedding improves recoverability but may reduce perceptual quality
- heavy geometric attacks can challenge peak detection and affine estimation

## 11) Troubleshooting

- Symptom: BER is near random (`~0.5`)
  - Check matching `key`, `patch_size`, `upsample_factor`, `tile_mode`.
  - Ensure message length and ECC settings are consistent.

- Symptom: Sync fails on extreme transforms
  - Increase image quality/resolution.
  - Tune `nms_size_ac`.
  - Try milder attack strength in benchmark.

- Symptom: Decode unstable across images
  - Increase `ecc_repetitions`.
  - Increase `upsample_factor` or adjust embedding strengths (`alpha_1`, `alpha_2`).

## 12) Reproducibility Guidelines

- Fix random seeds for message generation and attack noise.
- Keep model hyperparameters versioned with results.
- Report BER together with PSNR and runtime.
- Compare attacks on the same image subset and resolution.
