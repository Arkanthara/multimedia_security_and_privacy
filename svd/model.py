"""
SVD-based digital watermarking using Quantization Index Modulation (QIM).

Embeds a binary watermark into the singular values of an image's SVD
decomposition. Supports adaptive delta scaling, message repetition, and
majority-vote decoding.
"""

import numpy as np
from skimage.color import rgb2gray
from skimage.util import img_as_float64, img_as_ubyte


class WatermarkModel:
    """
    SVD watermarking via Quantization Index Modulation (QIM).

    The singular values of the host image are quantized to encode watermark
    bits. An adaptive delta schedule scales the quantization step with the
    magnitude of each singular value, preserving image quality while
    maintaining robustness.

    Parameters
    ----------
    psnr_threshold : float, optional
        Minimum acceptable PSNR (unused internally, kept for API compatibility).
    max_encode_time : float, optional
        Maximum encoding time in seconds (unused internally, kept for API
        compatibility).
    max_decode_time : float, optional
        Maximum decoding time in seconds (unused internally, kept for API
        compatibility).
    message_length : int, optional
        Number of bits in the watermark message.
    message_repetitions : int, optional
        Number of times the message is repeated for majority-vote robustness.
    quantization_delta : float, optional
        Base quantization step size :math:`\\delta_0`.  The effective step at
        position *i* is :math:`\\delta_i = \\delta_0 \\cdot k^i` where *k* is
        ``decay_factor``.
    decay_factor : float, optional
        Per-position decay multiplier *k* applied to ``quantization_delta``.
        Values < 1 shrink the step for later (smaller) singular values;
        ``1.0`` keeps the step constant.
    key : int, optional
        Seed for the pseudo-random position generator, ensuring identical
        embed/extract sequences.

    Notes
    -----
    Color images are converted to grayscale before watermarking; the returned
    watermarked image is also grayscale.

    The QIM rule used is:

    * bit = 1  →  quantize *s* to  :math:`[\\delta/2,\\ \\delta) \\pmod{\\delta}`
    * bit = 0  →  quantize *s* to  :math:`[0,\\ \\delta/2) \\pmod{\\delta}`
    """

    def __init__(
        self,
        psnr_threshold: float = 30.0,
        max_encode_time: float = 5.0,
        max_decode_time: float = 1.0,
        message_length: int = 32,
        message_repetitions: int = 1,
        quantization_delta: float = 10.0,
        decay_factor: float = 1.0,
        key: int = 42,
    ) -> None:
        self.psnr_threshold = psnr_threshold
        self.max_encode_time = max_encode_time
        self.max_decode_time = max_decode_time
        self.message_length = message_length
        self.message_repetitions = message_repetitions
        self.quantization_delta = quantization_delta
        self.decay_factor = decay_factor
        self.key = key

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def encode(self, image: np.ndarray, message: np.ndarray) -> np.ndarray:
        """
        Embed a binary watermark into *image*.

        Parameters
        ----------
        image : np.ndarray
            Host image, shape ``(H, W)`` or ``(H, W, C)``, dtype ``uint8``
            or float in ``[0, 1]``.
        message : np.ndarray
            1-D binary array of length ``message_length`` with values in
            ``{0, 1}``.

        Returns
        -------
        np.ndarray
            Watermarked grayscale image, same spatial dimensions as *image*,
            dtype ``uint8``.

        Raises
        ------
        ValueError
            If ``len(message) != message_length``.
        """
        if len(message) != self.message_length:
            raise ValueError(
                f"Expected message of length {self.message_length}, "
                f"got {len(message)}."
            )

        gray = self._to_gray_float(image)          # (H, W), float64 [0,1]
        payload = self._preprocess_message(message) # repeated bits

        # SVD of the host image
        U, s, Vt = np.linalg.svd(gray, full_matrices=False)

        # Embed watermark bits into singular values
        s_marked = self._embed_qim(s, payload)

        # Reconstruct and clip to [0, 1]
        watermarked = np.clip(U @ np.diag(s_marked) @ Vt, 0.0, 1.0)
        return img_as_ubyte(watermarked)

    def decode(self, image: np.ndarray) -> np.ndarray:
        """
        Extract the binary watermark from *image*.

        Parameters
        ----------
        image : np.ndarray
            Watermarked image, shape ``(H, W)`` or ``(H, W, C)``.

        Returns
        -------
        np.ndarray
            Recovered binary message, shape ``(message_length,)``, dtype
            ``int32``.
        """
        gray = self._to_gray_float(image)
        _, s, _ = np.linalg.svd(gray, full_matrices=False)

        total_bits = self.message_length * self.message_repetitions
        positions  = self._sample_positions(len(s), total_bits)

        # Extract one bit per position
        raw_bits = self._extract_qim(s, positions)

        return self._postprocess_message(raw_bits)

    # ------------------------------------------------------------------
    # Preprocessing / postprocessing helpers
    # ------------------------------------------------------------------

    def _preprocess_message(self, message: np.ndarray) -> np.ndarray:
        """
        Tile *message* ``message_repetitions`` times.

        Parameters
        ----------
        message : np.ndarray
            Binary array, shape ``(message_length,)``.

        Returns
        -------
        np.ndarray
            Tiled payload, shape ``(message_length * message_repetitions,)``.
        """
        return np.tile(message, self.message_repetitions).astype(np.int32)

    def _postprocess_message(self, raw_bits: np.ndarray) -> np.ndarray:
        """
        Recover the original message from repeated bits via majority vote.

        Parameters
        ----------
        raw_bits : np.ndarray
            Extracted bits, shape ``(message_length * message_repetitions,)``.

        Returns
        -------
        np.ndarray
            Decoded message, shape ``(message_length,)``, dtype ``int32``.
        """
        # Reshape into (message_length, message_repetitions) and majority vote
        reshaped = raw_bits.reshape(self.message_repetitions, self.message_length)
        # Sum over repetitions; threshold at half the repetition count
        votes = reshaped.sum(axis=0)
        return (votes > self.message_repetitions / 2).astype(np.int32)

    # ------------------------------------------------------------------
    # Pseudo-random position sampling
    # ------------------------------------------------------------------

    def _sample_positions(self, n_singular: int, n_bits: int) -> np.ndarray:
        """
        Sample *n_bits* unique positions from ``[0, n_singular)`` using
        the model's ``key`` for reproducibility.

        Parameters
        ----------
        n_singular : int
            Total number of singular values available.
        n_bits : int
            Number of positions to sample.

        Returns
        -------
        np.ndarray
            Sorted array of shape ``(n_bits,)`` with unique indices.

        Raises
        ------
        ValueError
            If *n_bits* exceeds *n_singular*.
        """
        if n_bits > n_singular:
            raise ValueError(
                f"Cannot embed {n_bits} bits into {n_singular} singular values."
            )
        rng = np.random.default_rng(self.key)
        # Sample without replacement, then sort to keep SVD ordering intact
        return np.sort(rng.choice(n_singular, size=n_bits, replace=False))

    # ------------------------------------------------------------------
    # Adaptive delta schedule
    # ------------------------------------------------------------------

    def _delta_schedule(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute the per-position effective quantization step.

        The step at index *i* is:

        .. math::
            \\delta_i = \\delta_0 \\cdot k^i

        where :math:`\\delta_0` is ``quantization_delta`` and *k* is
        ``decay_factor``.

        Parameters
        ----------
        positions : np.ndarray
            Sorted integer indices into the singular-value vector.

        Returns
        -------
        np.ndarray
            Array of effective deltas, same shape as *positions*.
        """
        exponents = positions.astype(np.float64)
        return self.quantization_delta * (self.decay_factor ** exponents)

    # ------------------------------------------------------------------
    # QIM embed / extract
    # ------------------------------------------------------------------

    def _embed_qim(self, s: np.ndarray, payload: np.ndarray) -> np.ndarray:
        """
        Quantize selected singular values to encode *payload* bits.

        For each (position, bit) pair the singular value is shifted into the
        appropriate half-interval of the quantization cell:

        * bit = 1  →  remainder in :math:`[\\delta/2, \\delta)`
        * bit = 0  →  remainder in :math:`[0, \\delta/2)`

        The shift is chosen so that the absolute change is minimal (nearest
        quantization center), which maximises PSNR. After quantization,
        values are clamped to ensure the *sorted order* of singular values
        is preserved (required for stable reconstruction).

        Parameters
        ----------
        s : np.ndarray
            Original singular values, shape ``(K,)``, sorted descending.
        payload : np.ndarray
            Binary array of bits to embed, shape ``(n_bits,)``.

        Returns
        -------
        np.ndarray
            Modified singular values with the same shape as *s*.
        """
        positions = self._sample_positions(len(s), len(payload))
        deltas    = self._delta_schedule(positions)

        s_marked = s.copy()

        for idx, (pos, bit, delta) in enumerate(zip(positions, payload, deltas)):
            val       = s_marked[pos]
            remainder = val % delta           # current position within cell

            if bit == 1:
                # Target: remainder in [delta/2, delta)
                if remainder < delta / 2:
                    # Shift up to nearest centre 3*delta/4
                    quantized = val - remainder + 3 * delta / 4
                else:
                    # Already in target half; snap to centre
                    quantized = val - remainder + 3 * delta / 4
            else:
                # Target: remainder in [0, delta/2)
                if remainder >= delta / 2:
                    # Shift down to nearest centre delta/4
                    quantized = val - remainder + delta / 4
                else:
                    quantized = val - remainder + delta / 4

            s_marked[pos] = quantized

        # ------------------------------------------------------------------
        # Preserve sorted order: clamp each value to the interval defined by
        # its immediate neighbours so the reconstruction U S Vt is stable.
        # ------------------------------------------------------------------
        for pos in positions:
            lo = s_marked[pos + 1] if pos + 1 < len(s_marked) else 0.0
            hi = s_marked[pos - 1] if pos - 1 >= 0          else np.inf
            s_marked[pos] = np.clip(s_marked[pos], lo, hi)

        return s_marked

    def _extract_qim(self, s: np.ndarray, positions: np.ndarray) -> np.ndarray:
        """
        Decode bits from singular values at *positions*.

        A bit is decoded as 1 if the remainder of the singular value divided
        by the effective delta falls in :math:`[\\delta/2, \\delta)`, and 0
        otherwise.

        Parameters
        ----------
        s : np.ndarray
            Singular values of the (possibly distorted) watermarked image.
        positions : np.ndarray
            Sorted array of embedding positions (output of
            ``_sample_positions``).

        Returns
        -------
        np.ndarray
            Extracted bits, shape ``(len(positions),)``, dtype ``int32``.
        """
        deltas     = self._delta_schedule(positions)
        remainders = s[positions] % deltas
        # bit = 1 iff remainder is in the upper half of the quantization cell
        return (remainders >= deltas / 2).astype(np.int32)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _to_gray_float(image: np.ndarray) -> np.ndarray:
        """
        Convert *image* to a float64 grayscale array in ``[0, 1]``.

        Parameters
        ----------
        image : np.ndarray
            Input image, ``uint8`` or float, grayscale or RGB/RGBA.

        Returns
        -------
        np.ndarray
            2-D float64 array, shape ``(H, W)``, values in ``[0, 1]``.
        """
        img = img_as_float64(image)
        if img.ndim == 3:
            img = rgb2gray(img)
        return img
