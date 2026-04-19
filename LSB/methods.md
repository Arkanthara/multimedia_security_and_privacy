You are an expert Python engineer specializing in high-performance numerical computing and image processing using NumPy.

Your task is in a first part to implement a **highly optimized, vectorized, and production-quality watermarking system** for a competition setting.

In a second part, you will implement a **jupyter notebook** that demonstrates the performance of your watermarking model on a variety of transformations and attacks, showcasing its robustness and efficiency.

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

## Project Structure

* You must create a notebook named `benchmark.ipynb` that demonstrates the performance of watermarking models on a variety of transformations and attacks.
* The structure of the project should be as follows:

```
.
├── lsb/
|   ├── utils.py (optional)
│   └── model.py
├── spatial/
│   ├── utils.py (optional)
│   └── model.py
├── utils/ (optional, for common utilities across methods)
├── benchmark.ipynb
├── pyproject.toml
└── img/
```

* The `img/` folder will contain any images needed for testing and demonstration in the notebook.
* Note that the `lsb/` folder is where you will implement the `WatermarkModel` class for the LSB method.
* Other methods (e.g., DCT, DWT) will be later implemented in separate folders with their own `model.py` files following the same structure and interface, so the benchmark notebook must be designed to easily import and test different models from different folders.
* For utility functions (e.g., patch management/synchronization), you can create a separate file (e.g., `utils.py`) in the method's folder, but ensure that the core encoding/decoding logic is implemented in `model.py`.
  They must use paths relative to the current file:

  ```python
  from pathlib import Path
  current_dir = Path(__file__).resolve().parent
  ```

## Problem Context

You are implementing in a first stage a watermarking model evaluated on:

* Robustness to transformations
* Visual quality (PSNR constraint)
* Runtime efficiency

You are implementing in a second stage a benchmark notebook that demonstrates the performance of the watermarking model on a variety of transformations and attacks, showcasing its robustness and efficiency.
The attacks and transformations to test include:
* Rotation
* Scaling
* Cropping
* Flipping
* Noise addition
* Compression (e.g., JPEG)
* Color space conversion (for color images)
* Blurring
* Sharpening
* Contrast adjustment
* Brightness adjustment
* Hue/Saturation adjustment
* Geometric distortions (e.g., perspective transform)
* Shifting
* Resizing
* Gamma correction
* And any other relevant transformations that can occur in real-world scenarios

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

Use `opencv-python` (or scikit-image if needed like for psnr calculation, but prefer `opencv-python` due to its performance) for any image processing needs (e.g., transformations, color space conversions, etc.) but ensure that the core encoding/decoding logic is implemented using **pure NumPy** for maximum performance.

I want two implementations of the `WatermarkModel` class: one for the LSB method and one for the spatial method (which will be implemented later). The benchmark notebook must be designed to easily import and test both models.
The lsb and spatial methods are very similar, except in some steps that will be described later.

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

Note that for the LSB method, the patch must not be converted to -1 and 1, but must stay in 0 and 1, and no alpha must be applied.

### Decoding Strategy

* Denoise the image and subtract the denoised image from the original to get the noise.
* In case of LSB, we can directly get the noise by getting the least significant bit of the image. Note that in this case, we must work in [0, 255] with uint8 image. The next steps are the same, except that we don't need to apply the decision rule on the noise, but we can directly get the bits by majority voting on number of times the bit is 1 or 0 across the repetitions.

For the next steps, we will test some transformations and keep the better results (can be determined by majority voting: a vote of 50/50 has less value than a clear majority):
    - no transform
    - vertical flip
    - horizontal flip since they are not managed by the log-polar + phase correlation synchronization.
In this way, even if the image is flipped and rotated, we can still retrieve the watermark.
If during all theses tests a test is successful, we can stop and return the result, otherwise we can return the result of the test with the highest majority voting metric (the absolute value of the sum of the bits across repetitions for instance).

* If synchronization is enabled (activated by default), use the sync pattern to align the noise with the original patch (using log-polar + phase correlation).
* Sum up the noise in the corresponding positions of the patch (using the same key-based randomness to get the positions) to get a value for each bit.
* Apply a decision rule (e.g., if the value is positive, decode as 1, else decode as 0) to retrieve the watermark bits.
* Apply majority voting to get the final message.
* (Optional) If error correction is used, apply the corresponding decoding to retrieve the original message. (will come later if we have time)
* Return the decoded message.

For synchronization (if enabled)(for both LSB and spatial methods):

* Regenerate the same patch using the same key-based randomness. (Note that here, we don't know the watermark bits, so we just regenerate the patch with the pseudo-random values, which will be used for synchronization).
* Less than 0 becomes 0, more than 0 becomes 1. (note that if LSB is used, the noise is already in binary values, so we can skip this step)
* Use log-polar + phase correlation to detect the sync pattern (which is simply the random pattern of the patch repeated accross the image) in the noise (rotation and scaling) (note that for phase correlation, the patch must be given as reference to detect changes applied to noise...).
* Return the inverse transformations to the noise to align it with the original patch (if necessary)

---

### Majority Voting

* If message is repeated:

  * For each bit position:

    * Sum the values across all repetitions
    * Get the sign of the sum to determine the bit value (positive = 1, negative = 0)
    * Metric used for majority voting can be the absolute value of the sum (the higher, the more confident the decision is)

### Noise Visibility Function

* If enabled, apply a noise visibility function to the patch before embedding to further improve visual quality (note that this is not compatible with the lsb method).
* The noise visibility function is defined like this:

  ```python
  # Compute NVF ( Noise Visibility Function )
  local_variance = compute_local_variance(image , WINDOW_SIZE)
  max_variance = np. max (local_variance)
  nvf = 1 / (1 + D * local_variance / max_variance)
  ```
  with D a parameter controlling the visibility of the noise and the local variance defined like this:
  ```python
  # Function to compute local variance
  def compute_local_variance(image: np.ndarray, window_size: int) -> np.ndarray:
    kernel = np.ones((window_size , window_size), np.float32) / (window_size** 2)
    mean_image = cv2.filter2D(image , -1, kernel)
    variance_image = cv2.filter2D(image ** 2, -1, kernel) - mean_image ** 2
    return variance_image
  ```
* If enabled, use like this the NVF to modulate the patch before embedding:

```python
# Weighted Stego Image
masked_watermark = (1 - nvf) * watermark * S_WEIGHTED_SCALED + nvf * watermark * S1_WEIGHTED_SCALED
stego_weighted = np.clip(image + masked_watermark , 0, 1)
```

### Recap of the differences between LSB and spatial methods:

| Aspect                     | LSB Method                               | Spatial Method                            |
|----------------------------|------------------------------------------|-------------------------------------------|
| Patch Values               | Binary (0 and 1)                         | Continuous (e.g., -1 and 1)               |
| Alpha Application          | No alpha applied                         | Alpha applied to control strength         |
| Noise Visibility Function  | Not compatible                           | Compatible                                |
| Encoding Strategy          | Directly embed bits in LSB               | Modulate pixel values with patch          |
| Decoding Strategy          | Extract bits from LSB                    | Denoise and apply decision rule           |
| Synchronization            | Optional (log-polar + phase correlation) | Optional (log-polar + phase correlation)  |
| Error Correction           | Optional (e.g., Turbo Codes, LDPC)       | Optional (e.g., Turbo Codes, LDPC)        |

---

## Testing and Benchmarking

The `benchmark.ipynb` notebook is composed of two main sections:
* Attacks and transformations demonstration
* Benchmarking and performance comparison

### Metrics

* Bit Error Rate (BER): Measures the percentage of bits that were incorrectly extracted compared to the original watermark. Lower is better.
* Peak Signal-to-Noise Ratio (PSNR): Measures the visual quality of the watermarked image compared to the original image. Higher is better.
* Runtime: Measure the time taken for encoding and decoding. Lower is better.

If already implemented in libraries, we can use existing functions to compute these metrics.

### Attacks and Transformations Demonstration

* Tests on a variety of transformations and attacks (as listed above)
* For each attack (including no attack), show the results clearly with fixed parameters (e.g., alpha, repeat, etc.) for each implementation (LSB and spatial methods actually...):
  * Show the original image, the original watermark, the attacked image, the target watermark, and the extracted watermark
  * Report the metrics (BER, PSNR, runtime)

### Benchmarking and Performance Comparison

* Systematically test a range of parameters (e.g., alpha values, repeat values, synchronization enabled/disabled, noise visibility function enabled/disabled, etc.) for each method (LSB and spatial methods actually...) across all selected attacks.
* For each configuration, compute the BER scores for all selected attacks, the PSNR score of the watermarked image, and the runtime for encoding and decoding.
* Store all results in a structured format (e.g., Pandas DataFrame) for easy comparison and analysis.
* Perform an overall performance comparison by calculating the mean BER score across all attacks for each configuration, and sort the results by this overall performance score to identify the best performing configurations.
* Clearly describe each configuration with its parameters and the method used (LSB or spatial method actually... Others will be added later...).
* Ensure that all images stored in the `img/` folder are used in the notebook for testing during the benchmarking phase. This will provide a comprehensive evaluation of the watermarking models under various imaging conditions.
* The benchmarking results should be presented in a clear and concise manner, such as tables or visualizations, to facilitate easy comparison of the different configurations and methods. This will allow for quick identification of the best performing configurations based on the overall performance score.
* The benchmarking process should be designed to be efficient and scalable, allowing for the addition of new methods, parameters, and attacks in the future without significant modifications to the existing codebase. This will ensure that the benchmarking framework remains relevant and useful as new watermarking techniques are developed and tested.
* The benchmarking results should be reproducible, with all random processes (e.g., patch generation, attack application) being deterministic based on the provided key. This will allow for consistent evaluation and comparison of different configurations and methods across multiple runs of the notebook.
* The benchmarking results should be comprehensive, covering a wide range of parameters and attacks to provide a thorough evaluation of the watermarking models. This will help to identify the strengths and weaknesses of each method under different conditions and guide future improvements in watermarking techniques.
* Avoid creating intermediate files (e.g., JSON or CSV) for storing results, and instead use in-memory data structures (e.g., Pandas DataFrame) to store and analyze the benchmarking results. This will improve efficiency and maintain a cleaner codebase while still allowing for comprehensive analysis and comparison of the different configurations and methods.
* Ensure that the benchmarking process is well-documented within the notebook, with clear explanations of the testing methodology, metrics used, and interpretation of results. This will enhance the readability and usability of the notebook for other researchers and practitioners in the field of watermarking.

## Performance Constraints

* MUST prioritize:

  * NumPy broadcasting
  * Vectorized memory access
  * Avoid Python loops unless absolutely necessary

* Encoding time must be less than `max_encode_time` seconds
* Decoding time must be less than `max_decode_time` seconds
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
* Use meaningful variable names

---

## Final Goal

Deliver two **state-of-the-art watermarking implementations** that are:

* Fast
* Robust
* Cleanly written
* As simple as possible while meeting all requirements
* Fully vectorized with NumPy
* Demonstrated in a comprehensive benchmark notebook showcasing their performance under various transformations and attacks.

Provide a clear and concise implementation that can be easily understood and used by other researchers and practitioners in the field of multimedia security and privacy.