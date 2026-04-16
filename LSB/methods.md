You are an expert Python engineer specializing in high-performance numerical computing and image processing using NumPy.

Your task is to implement a **highly optimized, vectorized, and production-quality watermarking system** for a competition setting.

## Strict Requirements

* You must create a code model.py that contains the implementation of the watermarking model 
* The class must be named exactly: `WatermarkModel`
* The implementation must be:

  * **Highly optimized (time-efficient)**
  * **Fully vectorized using NumPy (broadcasting prioritized)**
  * **Avoid Python loops as much as possible**
  * **Human-readable and well-structured**
  * **Fully documented using NumPy-style docstrings**
  * **As simple as possible while meeting all requirements**

---

## Problem Context

You are implementing a watermarking model evaluated on:

* Robustness to transformations
* Visual quality (PSNR constraint)
* Runtime efficiency

---

## Interface to Implement

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

## Model Design Requirements

The code must work on **grayscale and color images** (handle both cases seamlessly with all channels managed).

Note that on color images, the watermark must be able to be hidden either in all the channels or in the channel Y of the YCbCr color space (Must be an option that is configurable).

The image is represented as a NumPy array of shape `(H, W)` for grayscale or `(H, W, C)` for color, with pixel values in the range [0, 1].

(As we work on [0, 1], if for instance alpha is more than 1, we assume user think about values in [0, 255], so we will divide alpha by 255 to get the correct value in [0, 1] for our implementation)

Use `opencv-python` (or scikit-image if needed, but prefer `opencv-python` due to its performance) for any image processing needs (e.g., PSNR calculation, transformations, etc.) but ensure that the core encoding/decoding logic is implemented using **pure NumPy** for maximum performance.

### Key-based Randomness

* The class must accept a `key` parameter
* This key is used to **initialize a deterministic random seed**
* This ensures reproducibility of index selection

---

### Repetition for Robustness

* Include a parameter: `repeat: int`
* The watermark must be repeated `n` times
* The index list must scale accordingly

### Patch Generation (CRITICAL)

* Generate a patch of size 32x32 (or any configurable size) with pseudo-random 0 and 1 values (using the key-based randomness for reproducibility).
* Select random positions in the patch using the same key-based randomness to ensure reproducibility.
* Hide watermark bits in the selected positions (0 for 0, 1 for 1).
* Then, upsample the patch two times to 64x64 (or the corresponding size) (so each bit is repeated in a 2x2 block) to increase robustness. (note that actually, the patch is only composed of 0 and 1 values, so the upsampling is just to repeat each bit in a 2x2 block, which increases robustness)
* Finally, pad the patch to the size of the image.
* Return the image-sized patch as the final watermarked image to embed.
(note that here, the repeat parameter determines how many times the message is repeated in the patch...)

### Encoding Strategy

* Take the original image and the generated patch.
* (optional) If ECC is used, apply the corresponding algorithm to get the correct ECC before embedding it in the patch. Then, if repeat > 1, the message is encoded once whereas the ECC is encoded multiple times... (will come later if we have time... Will use Turbo Codes (prefered) or LDPC codes for error correction)
* Multiply the patch by 2 and substract 1 to get all values of the patch to be either -1 or 1.
* Make image + alpha * patch where alpha controls the strength of the watermark.
* Ensure no overflow/underflow (clip values if necessary)
* Return the watermarked image.

Note that if LSB is used (parameter to activate), the patch must not be converted to -1 and 1, but must stay in 0 and 1, and no alpha must be applied.

### Decoding Strategy

* Denoise the image and subtract the denoised image from the original to get the noise.
* In case of LSB, we can directly get the noise by getting the least significant bit of the image. Note that in this case, we must work in [0, 255] with uint8 image. The next steps are the same, except that we don't need to apply the decision rule on the noise, but we can directly get the bits by majority voting on number of times the bit is 1 or 0 across the repetitions.

For the next steps, we will test some transformations and keep the better results (can be determined by majority voting: a vote of 50/50 has less value than a clear majority):
    - no transform
    - vertical flip
    - horizontal flip since they are not managed by the log-polar + phase correlation synchronization.
In this way, even if the image is flipped and rotated, we can still retrieve the watermark.

* If synchronization is enabled (activated by default), use the sync pattern to align the noise with the original patch (using log-polar + phase correlation).
* Sum up the noise in the corresponding positions of the patch (using the same key-based randomness to get the positions) to get a value for each bit.
* Apply a decision rule (e.g., if the value is positive, decode as 1, else decode as 0) to retrieve the watermark bits.
* Apply majority voting to get the final message.
* (Optional) If error correction is used, apply the corresponding decoding to retrieve the original message. (will come later if we have time)
* Return the decoded message.

For synchronization (if enabled):

* Regenerate the same patch using the same key-based randomness. (Note that here, we don't know the watermark bits, so we just regenerate the patch with the pseudo-random values, which will be used for synchronization).
* Use log-polar + phase correlation to detect the sync pattern (which is simply the random pattern of the patch repeated accross the image) in the noise (rotation and scaling) (note that for phase correlation, the patch must be given as reference, so we can use the same patch generated with the same key-based randomness as the reference for phase correlation...).
* Return the inverse transformations to the noise to align it with the original patch (if necessary)

---

### Majority Voting

* If message is repeated:

  * For each bit position:

    * Sum the values across all repetitions
    * Get the sign of the sum to determine the bit value (positive = 1, negative = 0)
    * Metric used for majority voting can be the absolute value of the sum (the higher, the more confident the decision is)

---

## Performance Constraints

* MUST prioritize:

  * NumPy broadcasting
  * Vectorized memory access
  * Avoid Python loops unless absolutely necessary


---

## Documentation

* Every function must include **NumPy-style docstrings**
* Clearly describe:

  * Parameters
  * Returns
  * Behavior
  * Edge cases

---

## Code Quality

* Clean, modular, readable
* Use helper methods where appropriate
* Avoid unnecessary abstractions/functions
* Include comments explaining key optimizations

---

## Additional Notes

* If external files are needed:

  * Use paths relative to the current file:

```python
from pathlib import Path
current_dir = Path(__file__).resolve().parent
```

---

## Final Goal

Deliver a **state-of-the-art watermarking implementation** that is:

* Fast
* Robust
* Cleanly written
* As simple as possible while meeting all requirements
* Fully vectorized with NumPy