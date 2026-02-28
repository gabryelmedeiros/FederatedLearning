# -*- coding: utf-8 -*-
"""
Gradient/Weight Compression Methods for Federated Learning.

Implements:
  - MPGBP (Matching Pursuits with Generalized Bit Planes)
  - QSGD  (Quantized Stochastic Gradient Descent)
  - SignSGD (Sign-based compression)
  - No compression (identity baseline)

Each compressor follows the same interface:
    compressed = compressor(tensor, **kwargs)
    Returns a tensor of the same shape.
"""

import torch
import math


# =============================================================================
# MPGBP - Matching Pursuits with Generalized Bit Planes
# =============================================================================

def spt_count(x_hat: torch.Tensor, max_iter: int = 64) -> torch.Tensor:
    """
    Count the number of signed powers of two (SPTs) in the approximation.

    Each nonzero element is decomposed as a sum of powers of two;
    the total count across all elements is the representation cost.

    Args:
        x_hat:    Approximated vector.
        max_iter: Safety cap to avoid infinite loops from float rounding.

    Returns:
        Total SPT count (scalar tensor).
    """
    K = len(x_hat)
    count = torch.zeros(K)

    for idx in range(K):
        residue = torch.abs(x_hat[idx])
        iters = 0
        while residue > 1e-12 and iters < max_iter:
            bitplane = torch.floor(torch.log2(residue))
            residue = torch.abs(residue - 2 ** bitplane)
            count[idx] += 1
            iters += 1

    return torch.sum(count)


def mpgbp(x: torch.Tensor, M_max: int, P: int, epsilon: float = 1e-6) -> torch.Tensor:
    """
    MPGBP algorithm for vector quantization using signed powers of two.

    Iteratively builds a sparse approximation x_hat of the input x by
    adding scaled codewords (sparse vectors with entries in {-1, 0, +1})
    where the scaling factor is quantized to the nearest power of two.

    Args:
        x:     Input vector (any shape, will be flattened internally).
        M_max: Maximum total number of SPTs allowed in the approximation.
        P:     Number of nonzero entries per codeword.
        epsilon: Early stopping threshold on residual norm.

    Returns:
        Approximation x_hat, same shape and dtype as input.
    """
    orig_shape = x.shape
    dtype = x.dtype
    x_vec = x.detach().reshape(-1).float()
    N = len(x_vec)

    x_hat = torch.zeros(N)
    residue = x_vec.clone()
    M = 0

    while M < M_max:
        # Build sparse codeword from top-P residual positions
        codeword = torch.zeros(N)
        idx_sort = torch.argsort(-torch.abs(residue))

        for i in range(min(P, N)):
            pos = idx_sort[i].item()
            codeword[pos] = torch.sign(residue[pos])

        # Skip if codeword is zero (residue is essentially zero)
        codeword_norm = torch.norm(codeword, p=2)
        if codeword_norm < 1e-12:
            break

        # Optimal step: project residue onto normalized codeword direction
        codeword_normalized = codeword / codeword_norm
        alpha = torch.dot(residue, codeword_normalized) / codeword_norm

        # Quantize alpha to nearest power of two (MPGBP rule)
        if alpha <= 0:
            break
        power = -torch.ceil(torch.log2(3.0 / (4.0 * alpha)))
        spt = 2.0 ** power

        # Update approximation and residue
        residue = residue - spt * codeword
        x_hat = x_hat + spt * codeword

        # Check SPT budget
        M = spt_count(x_hat).item()

        # Check convergence
        if torch.norm(residue, p=2) <= epsilon:
            break

    return x_hat.reshape(orig_shape).to(dtype)


def compress_mpgbp(tensor: torch.Tensor, M_max: int = 16,
                   mode: str = "C", epsilon: float = 1e-6) -> torch.Tensor:
    """
    Apply MPGBP compression to a tensor (gradient or weight).

    Modes:
        "V" - Vectorized: treat entire tensor as one vector (for gradients)
        "C" - Per-channel/row: quantize each row independently
        "R" - Per-element-group: (reserved for future use)

    Args:
        tensor:  Input tensor to compress.
        M_max:   Maximum SPTs per quantization unit.
        mode:    Quantization granularity.
        epsilon: Convergence threshold.

    Returns:
        Compressed tensor, same shape.
    """
    if mode == "V":
        # Flatten entire tensor, quantize as one vector
        flat = tensor.detach().reshape(-1)
        N = flat.numel()
        P = max(1, math.ceil(math.sqrt(N)))
        return mpgbp(flat, M_max=M_max, P=P, epsilon=epsilon).reshape(tensor.shape)

    elif mode == "C":
        # Quantize each row (channel/neuron) independently
        result = torch.zeros_like(tensor)
        if tensor.dim() == 1:
            N = tensor.numel()
            P = max(1, math.ceil(math.sqrt(N)))
            result = mpgbp(tensor, M_max=M_max, P=P, epsilon=epsilon)
        else:
            # Iterate over first dimension
            for i in range(tensor.shape[0]):
                row = tensor[i]
                N = row.numel()
                P = max(1, math.ceil(math.sqrt(N)))
                result[i] = mpgbp(row, M_max=M_max, P=P, epsilon=epsilon)
        return result

    else:
        raise ValueError(f"Unknown MPGBP mode: {mode}. Use 'V' or 'C'.")


# =============================================================================
# QSGD - Quantized Stochastic Gradient Descent (Alistarh et al., 2017)
# =============================================================================

def compress_qsgd(tensor: torch.Tensor, n_bits: int = 2) -> torch.Tensor:
    """
    QSGD: stochastic quantization of a gradient vector.

    Each component v_i is quantized to one of s = 2^n_bits levels,
    using randomized rounding that is unbiased: E[Q(v)] = v.

    The quantization is: Q(v_i) = ||v|| * sign(v_i) * xi_i
    where xi_i = floor(|v_i|/||v|| * s) / s  with probability p,
    and ceil(...)/s with probability 1-p (to maintain unbiasedness).

    Args:
        tensor: Input gradient tensor.
        n_bits: Number of quantization bits (s = 2^n_bits levels).

    Returns:
        Quantized tensor, same shape.
    """
    s = 2 ** n_bits
    flat = tensor.detach().reshape(-1).float()

    norm = torch.norm(flat, p=2)
    if norm < 1e-12:
        return torch.zeros_like(tensor)

    # Normalize by L2 norm
    normalized = torch.abs(flat) / norm

    # Stochastic rounding
    scaled = normalized * s
    floor_val = torch.floor(scaled)
    prob = scaled - floor_val  # probability of rounding up

    # Bernoulli rounding
    rand = torch.rand_like(prob)
    quantized_level = torch.where(rand < prob, floor_val + 1, floor_val)

    # Reconstruct: ||v|| * sign(v) * (quantized_level / s)
    signs = torch.sign(flat)
    result = norm * signs * (quantized_level / s)

    return result.reshape(tensor.shape).to(tensor.dtype)


# =============================================================================
# SignSGD (Bernstein et al., 2018)
# =============================================================================

def compress_signsgd(tensor: torch.Tensor) -> torch.Tensor:
    """
    SignSGD: transmit only the sign of each gradient component,
    scaled by the mean absolute value (for magnitude preservation).

    This is 1-bit compression: each element becomes +scale or -scale.

    Args:
        tensor: Input gradient tensor.

    Returns:
        Sign-compressed tensor, same shape.
    """
    flat = tensor.detach().reshape(-1).float()
    scale = torch.mean(torch.abs(flat))
    signs = torch.sign(flat)
    # Replace zeros with +1 (convention)
    signs[signs == 0] = 1.0
    result = scale * signs
    return result.reshape(tensor.shape).to(tensor.dtype)


# =============================================================================
# No compression (baseline)
# =============================================================================

def compress_none(tensor: torch.Tensor) -> torch.Tensor:
    """Identity: no compression applied."""
    return tensor.clone()


# =============================================================================
# Compression cost estimation (bits transmitted)
# =============================================================================

def estimate_bits(tensor: torch.Tensor, method: str, **kwargs) -> int:
    """
    Estimate the number of bits required to transmit a compressed tensor.

    Args:
        tensor: The original (uncompressed) tensor.
        method: Compression method name.
        **kwargs: Method-specific parameters.

    Returns:
        Estimated number of bits.
    """
    numel = tensor.numel()

    if method == "none":
        # Full precision float32
        return numel * 32

    elif method == "signsgd":
        # 1 bit per element + 32 bits for the scale factor
        return numel * 1 + 32

    elif method == "qsgd":
        n_bits = kwargs.get("n_bits", 2)
        # n_bits per element + 32 bits for the norm
        return numel * n_bits + 32

    elif method == "mpgbp":
        # For MPGBP: each SPT needs log2(N) bits for position + 1 bit for sign
        # + exponent bits for the power of two.
        # Simplified estimate: M_max * (ceil(log2(N)) + 1 + 8) per quantization unit
        M_max = kwargs.get("M_max", 16)
        mode = kwargs.get("mode", "V")
        if mode == "V":
            N = numel
            bits_per_spt = math.ceil(math.log2(max(N, 2))) + 1 + 8
            return M_max * bits_per_spt
        elif mode == "C":
            total = 0
            if tensor.dim() == 1:
                N = numel
                bits_per_spt = math.ceil(math.log2(max(N, 2))) + 1 + 8
                total = M_max * bits_per_spt
            else:
                for i in range(tensor.shape[0]):
                    N = tensor[i].numel()
                    bits_per_spt = math.ceil(math.log2(max(N, 2))) + 1 + 8
                    total += M_max * bits_per_spt
            return total
        else:
            return numel * 32  # fallback

    return numel * 32  # default: uncompressed


# =============================================================================
# Unified interface
# =============================================================================

def get_compressor(method: str, **kwargs):
    """
    Factory function returning a (compress_fn, bits_fn) tuple.

    Usage:
        compress_fn, bits_fn = get_compressor("qsgd", n_bits=4)
        compressed_grad = compress_fn(gradient)
        bits_used = bits_fn(gradient)
    """
    if method == "none":
        return compress_none, lambda t: estimate_bits(t, "none")

    elif method == "signsgd":
        return compress_signsgd, lambda t: estimate_bits(t, "signsgd")

    elif method == "qsgd":
        n_bits = kwargs.get("n_bits", 2)
        return (
            lambda t: compress_qsgd(t, n_bits=n_bits),
            lambda t: estimate_bits(t, "qsgd", n_bits=n_bits),
        )

    elif method == "mpgbp":
        M_max = kwargs.get("M_max", 16)
        mode = kwargs.get("mode", "V")
        epsilon = kwargs.get("epsilon", 1e-6)
        return (
            lambda t: compress_mpgbp(t, M_max=M_max, mode=mode, epsilon=epsilon),
            lambda t: estimate_bits(t, "mpgbp", M_max=M_max, mode=mode),
        )

    else:
        raise ValueError(f"Unknown compression method: {method}")
