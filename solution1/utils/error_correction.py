"""
utils/error_correction.py
=========================
LDPC-based error correction utilities for spatial watermarking.

This module uses repetition codes (a special class of LDPC codes) together
with the belief-propagation (BP) decoder from the ``ldpc`` library.

For a message of *k* bits and a repetition factor *r*, the codeword has
length *k × r*.  Each original bit occupies a contiguous block of *r*
positions:

    message  : [b0, b1, ..., b_{k-1}]
    codeword : [b0, b0, ..., b0, b1, b1, ..., b1, ..., b_{k-1}, ...]
                |------- r ---------|

Encoding uses the LDPC generator matrix (``G^T · b mod 2``).
Decoding uses BP received-vector decoding (corrects the corrupted codeword).

References
----------
.. [1] Gallager, R. G. (1962). Low-density parity-check codes.
       IRE Trans. Inf. Theory, 8(1), 21–28.
.. [2] Roffe, J. (2023). ldpc — Python library for LDPC codes.
       https://software.roffe.eu/ldpc/
"""

from __future__ import annotations

import numpy as np
from ldpc import BpDecoder
from ldpc.codes import rep_code
from ldpc.code_util import construct_generator_matrix
from scipy.sparse import block_diag as sparse_block_diag


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _build_block_diagonal_H(n_repetitions: int, k: int) -> np.ndarray:
    """
    Build the dense parity-check matrix *H* for *k* independent repetition codes.

    Each single-bit repetition code of length *r* has H_single = rep_code(r),
    shape (r-1, r).  The full matrix is block-diagonal over *k* such blocks:

        shape(H) = (k * (r-1),  k * r)

    Parameters
    ----------
    n_repetitions : int
        Repetition factor ``r``.
    k : int
        Number of independent message bits.

    Returns
    -------
    H : ndarray of shape (k*(r-1), k*r), dtype uint8
    """
    H_single = rep_code(n_repetitions)                     # (r-1, r) sparse
    H_block  = sparse_block_diag([H_single] * k)           # (k*(r-1), k*r) sparse
    return np.asarray(H_block.toarray(), dtype=np.uint8)


def _build_block_diagonal_G(n_repetitions: int, k: int):
    """
    Build the block-diagonal generator matrix for *k* independent repetition
    codes.

    Each single-bit repetition code of length *r* has generator matrix
    G_single of shape (1, r), equivalent to ``[1, 1, ..., 1]``.  The full
    matrix is block-diagonal:

        shape(G_full) = (k, k * r)

    Parameters
    ----------
    n_repetitions : int
    k : int

    Returns
    -------
    G_full : scipy sparse matrix of shape (k, k*r), dtype uint8
    """
    H_single = rep_code(n_repetitions)                     # (r-1, r) sparse
    G_single = construct_generator_matrix(H_single)         # (1, r)   sparse
    return sparse_block_diag([G_single] * k)               # (k, k*r) sparse


# ---------------------------------------------------------------------------
# Encoding — via LDPC generator matrix
# ---------------------------------------------------------------------------

def repetition_encode(bits: np.ndarray, n_repetitions: int) -> np.ndarray:
    """
    Encode *bits* using a repetition code via the LDPC generator matrix.

    The generator matrix ``G`` (shape ``(k, k*r)``) is built block-diagonally
    from ``k`` independent single-bit repetition codes of length ``r``.
    The codeword is computed as::

        codeword = G^T · bits  (mod 2)

    For the repetition code this reduces to ``np.repeat``, but the
    implementation uses the ldpc generator matrix explicitly.

    Parameters
    ----------
    bits : ndarray of shape (k,), dtype uint8
        Message bits in {0, 1}.
    n_repetitions : int
        Number of copies per bit (``r``).  Must be ≥ 1.

    Returns
    -------
    codeword : ndarray of shape (k * n_repetitions,), dtype uint8
        Encoded bit sequence.
    """
    bits = bits.astype(np.uint8)
    if n_repetitions == 1:
        return bits.copy()
    k = len(bits)
    if k == 0:
        return np.array([], dtype=np.uint8)

    G_full   = _build_block_diagonal_G(n_repetitions, k)   # (k, k*r) sparse
    codeword = G_full.T.dot(bits) % 2                      # (k*r,) array
    return np.asarray(codeword).flatten().astype(np.uint8)


# ---------------------------------------------------------------------------
# Decoding — hard-decision majority vote (fast fallback)
# ---------------------------------------------------------------------------

def repetition_decode_majority(bits: np.ndarray, n_repetitions: int) -> np.ndarray:
    """
    Decode a repetition-coded sequence using majority vote.

    This is the maximum-likelihood decoder for a BSC (binary symmetric
    channel) and is optimal when all repetition positions are equally noisy.

    Parameters
    ----------
    bits : ndarray of shape (k * n_repetitions,), dtype uint8
        (Possibly corrupted) codeword bits in {0, 1}.
    n_repetitions : int
        Repetition factor used during encoding.

    Returns
    -------
    message : ndarray of shape (k,), dtype uint8
        Decoded message bits in {0, 1}.
    """
    k = len(bits) // n_repetitions
    return (
        bits[: k * n_repetitions]
        .reshape(k, n_repetitions)
        .mean(axis=1)
        > 0.5
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Decoding — LDPC belief propagation (preferred)
# ---------------------------------------------------------------------------

def repetition_decode_ldpc(
    bits: np.ndarray,
    n_repetitions: int,
    channel_error_rate: float = 0.1,
    max_iter: int = 100,
    bp_method: str = "product_sum",
) -> np.ndarray:
    """
    Decode a repetition-coded sequence using LDPC belief-propagation.

    The decoder performs *received-vector decoding*: given a noisy codeword
    ``received``, ``BpDecoder.decode(received)`` returns the corrected
    codeword ``c``.  The original message bit for block ``i`` is then
    recovered as ``c[i * r]`` (the first position of each repetition block).

    The parity-check matrix ``H`` is block-diagonal, with one
    ``(r-1) × r`` repetition-code block per message bit.

    Parameters
    ----------
    bits : ndarray of shape (k * n_repetitions,), dtype uint8
        (Possibly corrupted) received codeword bits in {0, 1}.
    n_repetitions : int
        Repetition factor used during encoding.
    channel_error_rate : float, optional
        Expected fraction of flipped bits used to initialise BP LLRs.
    max_iter : int, optional
        Maximum number of BP iterations.
    bp_method : {"product_sum", "min_sum"}, optional
        Belief-propagation variant.

    Returns
    -------
    message : ndarray of shape (k,), dtype uint8
        Decoded message bits in {0, 1}.

    Notes
    -----
    Falls back to :func:`repetition_decode_majority` when ``n_repetitions``
    is 1, or if the BP decoder raises an exception.
    """
    bits = np.asarray(bits, dtype=np.uint8)
    if n_repetitions == 1:
        return bits.copy()

    k = len(bits) // n_repetitions
    if k == 0:
        return np.array([], dtype=np.uint8)

    received = bits[: k * n_repetitions]

    try:
        H = _build_block_diagonal_H(n_repetitions, k)      # (k*(r-1), k*r)

        decoder = BpDecoder(
            H,
            error_rate=channel_error_rate,
            max_iter=max_iter,
            bp_method=bp_method,
        )

        # Received-vector decoding: returns the corrected codeword
        corrected = decoder.decode(received)                # (k*r,)

        # For each repetition block the first position holds the message bit
        return corrected.reshape(k, n_repetitions)[:, 0].astype(np.uint8)

    except Exception:
        # Graceful fallback — majority vote is always robust
        return repetition_decode_majority(bits, n_repetitions)
