You are an expert Python engineer specializing in high-performance numerical computing and image processing using NumPy.

Your task is to implement a **highly optimized, vectorized, and production-quality watermarking system** for a competition setting.

## ⚠️ Strict Requirements

* The output must be a **single Python file named `model.py`** (for additionnal files, see instructions at Additional notes section) 
* The class must be named exactly: `WatermarkModel`
* The implementation must be:

  * **Highly optimized (time-efficient)**
  * **Fully vectorized using NumPy (broadcasting prioritized)**
  * **Avoid Python loops as much as possible**
  * **Human-readable and well-structured**
  * **Fully documented using NumPy-style docstrings**

---

## 🧠 Problem Context

You are implementing a watermarking model evaluated on:

* Robustness to transformations
* Visual quality (PSNR constraint)
* Runtime efficiency

---

## 🧩 Interface to Implement

Your class must follow this contract:

```python
def encode(self, image: np.ndarray, watermark: np.ndarray) -> np.ndarray: 
    """Embed a binary watermark into the image."""

def decode(self, watermarked_or_attacked_image: np.ndarray) -> np.ndarray: 
    """Extract the binary watermark from the image."""
```

Constructor must be configurable with:

* `message_length: int = 32`
* `psnr_threshold: float = 30.0`
* `max_encode_time: float = 5.0`
* `max_decode_time: float = 1.0`

---

## 🔐 Model Design Requirements

The code must work on **grayscale and color images** (handle both cases seamlessly with all channels managed).

The image is represented as a NumPy array of shape `(H, W)` for grayscale or `(H, W, C)` for color, with pixel values in the range [0, 255].

Use `opencv-python` for any image processing needs (e.g., PSNR calculation, transformations, etc.) but ensure that the core encoding/decoding logic is implemented using **pure NumPy** for maximum performance.

### 1. Key-based Randomness

* The class must accept a `key` parameter
* This key is used to **initialize a deterministic random seed**
* This ensures reproducibility of index selection

---

### 2. Index Generation (CRITICAL)

* Generate a list of pixel indices where the watermark will be embedded
* Must support:

  * Either **flat indexing** OR **multi-dimensional indexing**
  * Choose the fastest approach experimentally or heuristically
* Use NumPy vectorization (no loops)
* Must favor **decoding efficiency** more than encoding efficiency (since decoding is more time-sensitive)
* Note that in case of multi-channel images, the index generation must allow to hide watermark accross all channels.

---

### 3. Repetition for Robustness

* Include a parameter: `repeat: int`
* The watermark OR error correction code (if enabled) must be repeated `n` times
* The index list must scale accordingly

---

### 4. Error Correction Mode (Optional)

* Include a boolean parameter: `use_error_correction`
* If enabled:

  * Create an error-correcting code using an **efficient error-correcting code** (e.g., CRC, LDPC, Reed-Solomon, BCH or combination of them like CRC + LDPC etc.)
  * Repeat the error correction code `n` times BUT NOT the original message (to save space and increase robustness) (i.e., the original message is embedded once, but the ECC is repeated to protect it)
  * Allow two modes:
    * **Single ECC**: Only one algorithm (e.g., CRC, LDPC, ...)
    * **Combined ECC**: Multiple algorithms combined (e.g., CRC + LDPC, BCH + Reed-Solomon, etc.) for enhanced robustness
* Use **fast and reliable libraries** for encoding/decoding (e.g., `ldpc`, `bchlib`, `reedsolomon`, `crc`, etc.)

---

## 🧪 Encoding Strategy

* Flatten or reshape the image appropriately for fast access

* For each selected index:

  * If bit = 1 → pixel += alpha
  * If bit = 0 → pixel -= alpha

* `alpha` must be a configurable parameter of the class 

* Ensure:

  * No overflow/underflow (clip values if necessary)
  * PSNR constraint is respected

* Must be implemented using:

  * NumPy broadcasting
  * Vectorized operations

---

## 🔍 Decoding Strategy

1. Regenerate the same index list using the same key

2. For each index:

   * Extract a **3×3 neighborhood** (Avoid errors due to unmanaged cases where pixels are at the border of the image ! Strategy can be to use symmetric padding...) 
   * Compute either:

     * Mean OR
     * Median (choose the most robust/efficient)
     * Note that the center pixel must not be included in the statistic computation (to avoid biasing the decision)
   * Make it in a broadcasted way for all indices at once

3. Decision rule:

   * If center pixel < local statistic → decode as 0
   * Else → decode as 1

4. Use full vectorization for:

   * Neighborhood extraction
   * Statistic computation
   * Decision making

---

## 🗳️ Majority Voting

* If message is repeated:

  * For each bit position:

    * Aggregate all repetitions
    * Select majority value

* If error correction is used:

  * Apply majority voting on error correction code
  * Then retrieve the original message using ECC

---

## ⚡ Performance Constraints

* MUST prioritize:

  * NumPy broadcasting
  * Vectorized memory access
  * Avoid Python loops unless absolutely necessary
* Consider:

  * `np.take`, `np.put`, `np.ravel`, `np.reshape`
  * Stride tricks or efficient neighborhood extraction
* Code must be **the most fast and optimized using proper numpy operations to meet runtime constraints** 

---

## 📚 Documentation

* Every function must include **NumPy-style docstrings**
* Clearly describe:

  * Parameters
  * Returns
  * Behavior
  * Edge cases

---

## 🧼 Code Quality

* Clean, modular, readable
* Use helper methods where appropriate
* Avoid unnecessary abstractions/functions
* Include comments explaining key optimizations

---

## 📦 Additional Notes

* If external files are needed:

  * Use paths relative to the current file:

```python
from pathlib import Path
current_dir = Path(__file__).resolve().parent
```

---

## 🎯 Final Goal

Deliver a **state-of-the-art watermarking implementation** that is:

* Fast
* Robust
* Cleanly written
* Fully vectorized with NumPy


Heyyy !!! I have a working implementation of watermark embedding inside image.
Actually, it's not robust to rotations, and I want to make it robust to that.

To make the watermarking robust to rotations, I want that you add the option to test all rotations (from 0 to 360 degrees) according to a given step (e.g., every 15 degrees) during the decoding process...
I want that if some result is found before the end of the brute force rotations, the decoding process can stop immediately and return the result.

The code must:
* Be fully vectorized using NumPy
* Avoid Python loops as much as possible
* Be optimized for performance
* Be clean and well-structured
* Be fully documented using NumPy-style docstrings
* Be the simplest possible implementation to achieve the required functionality


Heyyy !!! This is a working implementation of watermark embedding inside image.

I want that you modify it: instead of computing a list of index, I prefer that you compute a mask of the same shape as the image (W, H)...

Heyyy !!! I have a matrix of size H x W.

I want that you create randomly a binary mask of the same shape (H, W) with a given number of True values (e.g., 104).

This mask will be a boolean array where True values indicate the pixels where the watermark will be embedded, and False values indicate the pixels that will remain unchanged.
For each channel, change the mask accordingly to hide the watermark across all channels.
Then, you must compute for a given mask a new mask that is a boolean array with True for neighborhood pixels of the original mask (i.e., for each True pixel in the original mask, the corresponding neighborhood pixels in the new mask will also be set to True), and False for the other pixels and the pixels of interest (i.e., the original True pixels) that will be used for the decision in the decoding process to avoid biasing the decision (i.e., the center pixel of the neighborhood must not be included in the statistic computation for the mean, only for the final comparison).

The code must be:
* Fully vectorized using NumPy
* Avoid Python loops as much as possible
* Optimized for performance
* Clean and well-structured
* Fully documented using NumPy-style docstrings
* The simplest possible implementation to achieve the required functionality

Heyyy !!! I have a matrix of size H x W.

I want that you create randomly (from a given seed) a binary mask of the same shape (H, W) with a given number of True values (e.g., 104).

Then, you must compute for a given mask a new mask that is a boolean array with True for neighborhood pixels of the original mask (i.e., for each True pixel in the original mask, the corresponding neighborhood pixels in the new mask will also be set to True), and False for the other pixels and the pixels of interest (i.e., the original True pixels) that will be used for the decision in the decoding process to avoid biasing the decision (i.e., the center pixel of the neighborhood must not be included in the statistic computation for the mean, only for the final comparison).

Then, I want a very efficient function that compute the impact of a rotation on the mask (i.e., compute the new mask after applying the rotation to the original mask... Mask must always be a boolean array with shape (H, W)) in a fully vectorized way using NumPy.
When the rotation has been applied, compute also the neighborhood mask for the new mask. (note that the neighborhood mask can be computed once and then rotated as well, instead of recomputing it from scratch for each rotation, depending on the efficiency of the rotation operation).

Note that the neighborhood can be of size 3x3, 5x5, etc. and the rotation step can be of 15 degrees, 30 degrees, 31.23 degrees, etc. depending on the configuration of the model.

You must produce a notebook that implement different method of mask generation and rotation, and compare their performance to choose the best one for the final implementation in the model.

The code must include plots of results and performance metrics to justify the choice of the final method.
Note that I want at least one method with rotation of neighborhood mask and one method with rotation of the original mask and then computation of the neighborhood mask for each rotation.

The code must be:
* Fully vectorized using NumPy
* Avoid Python loops as much as possible
* Optimized for performance
* Clean and well-structured
* Fully documented using NumPy-style docstrings
* The simplest possible implementation to achieve the required functionality