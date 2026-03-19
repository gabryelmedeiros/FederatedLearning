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


def extract_exponents(x_hat: torch.Tensor, max_iter: int = 64) -> list:
    """
    Extract all power-of-two exponents from the SPT representation of x_hat.

    Each nonzero element is decomposed into powers of two; the exponent
    (floor of log2 of the absolute value) is recorded for each term.
    The maximum absolute exponent determines the word length needed for
    hardware implementation (see Fig. 6 in da Silva et al., 2014).

    Args:
        x_hat:    Approximated vector (1-D).
        max_iter: Safety cap per element.

    Returns:
        List of integer exponents from all elements.
    """
    exponents = []
    for idx in range(len(x_hat)):
        residue = torch.abs(x_hat[idx])
        iters = 0
        while residue > 1e-12 and iters < max_iter:
            exp = torch.floor(torch.log2(residue))
            exponents.append(int(exp.item()))
            residue = torch.abs(residue - 2 ** exp)
            iters += 1
    return exponents


def mpgbp(x: torch.Tensor, M_max: int, P: int, epsilon: float = 1e-6,
          return_metadata: bool = False):
    """
    MPGBP algorithm for vector quantization using signed powers of two.

    Iteratively builds a sparse approximation x_hat of the input x by
    adding scaled codewords (sparse vectors with entries in {-1, 0, +1})
    where the scaling factor is quantized to the nearest power of two.

    Args:
        x:               Input vector (any shape, will be flattened internally).
        M_max:           Maximum total number of SPTs allowed in the approximation.
        P:               Number of nonzero entries per codeword.
        epsilon:         Early stopping threshold on residual norm.
        return_metadata: If True, return (x_hat, metadata_dict) instead of x_hat.

    Returns:
        x_hat (torch.Tensor) if return_metadata=False.
        (x_hat, dict) if return_metadata=True, where dict contains:
            "total_spts"      : int   — total SPTs in x_hat (from spt_count)
            "max_exponent"    : int   — max |exponent| across all SPT terms
            "num_iterations"  : int   — number of MPGBP iterations run
            "exponents_used"  : list  — per-iteration exponent (k_m values)
            "residual_norm"   : float — final ||residue||_2
    """
    orig_shape = x.shape
    dtype = x.dtype
    x_vec = x.detach().reshape(-1).float()
    N = len(x_vec)

    x_hat = torch.zeros(N)
    residue = x_vec.clone()
    M = 0
    iteration_exponents = []
    num_iterations = 0

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

        # Track iteration metadata
        iteration_exponents.append(int(power.item()))
        num_iterations += 1

        # Update approximation and residue
        residue = residue - spt * codeword
        x_hat = x_hat + spt * codeword

        # Check SPT budget
        M = spt_count(x_hat).item()

        # Check convergence
        if torch.norm(residue, p=2) <= epsilon:
            break

    result = x_hat.reshape(orig_shape).to(dtype)

    if return_metadata:
        total_spts = int(spt_count(x_hat).item())
        all_exponents = extract_exponents(x_hat)
        max_exp = max(abs(e) for e in all_exponents) if all_exponents else 0
        metadata = {
            "total_spts":     total_spts,
            "max_exponent":   max_exp,
            "num_iterations": num_iterations,
            "exponents_used": iteration_exponents,
            "residual_norm":  torch.norm(residue, p=2).item(),
        }
        return result, metadata

    return result


def compress_mpgbp(tensor: torch.Tensor, M_max: int = None,
                   multiplier: float = 1.0, mode: str = "R",
                   epsilon: float = 1e-6,
                   return_metadata: bool = False):
    """
    Apply MPGBP compression to a tensor (gradient or weight).

    Modes:
        "S" - Per-scalar:  each element quantized independently (P=1, M_max=1)
        "R" - Per-row:     each row (first dimension) quantized independently
        "L" - Per-layer:   entire tensor flattened into one vector
        "N" - Per-network: handled outside via apply_mpgbp(); not valid here

    Args:
        tensor:          Input tensor to compress.
        M_max:           Maximum SPTs per quantization unit. If given, used directly
                         (backwards compatible). If None, computed from multiplier.
        multiplier:      Budget multiplier for automatic M_max scaling:
                           R mode:  M_max_row   = ceil(multiplier * K / 2)
                           L mode:  M_max_layer = ceil(multiplier * K * R / 4)
                           S mode:  always M_max=1 regardless of multiplier
        mode:            Quantization granularity ("S", "R", or "L").
        epsilon:         Convergence threshold.
        return_metadata: If True, also return aggregated metadata dict.

    Returns:
        compressed tensor if return_metadata=False.
        (compressed tensor, metadata dict) if return_metadata=True.
        Metadata keys: total_spts, max_exponent, num_iterations, residual_norm.
    """
    def _empty_meta():
        return {"total_spts": 0, "max_exponent": 0,
                "num_iterations": 0, "residual_norm": 0.0}

    def _merge_meta(agg, meta):
        agg["total_spts"]     += meta["total_spts"]
        agg["max_exponent"]    = max(agg["max_exponent"], meta["max_exponent"])
        agg["num_iterations"] += meta["num_iterations"]
        agg["residual_norm"]  += meta["residual_norm"]

    if mode == "S":
        m = M_max if M_max is not None else 1
        flat = tensor.detach().reshape(-1).float()
        result = torch.zeros_like(flat)
        agg = _empty_meta()
        for i in range(flat.numel()):
            if return_metadata:
                val, meta = mpgbp(flat[i:i + 1], M_max=m, P=1,
                                  epsilon=epsilon, return_metadata=True)
                result[i] = val[0]
                _merge_meta(agg, meta)
            else:
                result[i] = mpgbp(flat[i:i + 1], M_max=m, P=1,
                                  epsilon=epsilon)[0]
        out = result.reshape(tensor.shape).to(tensor.dtype)
        return (out, agg) if return_metadata else out

    elif mode == "R":
        result = torch.zeros_like(tensor)
        agg = _empty_meta()
        if tensor.dim() == 1:
            K = tensor.numel()
            P = max(1, math.ceil(math.sqrt(K)))
            m = M_max if M_max is not None else max(1, math.ceil(multiplier * K / 2))
            if return_metadata:
                result, meta = mpgbp(tensor, M_max=m, P=P,
                                     epsilon=epsilon, return_metadata=True)
                _merge_meta(agg, meta)
            else:
                result = mpgbp(tensor, M_max=m, P=P, epsilon=epsilon)
        else:
            for i in range(tensor.shape[0]):
                row = tensor[i]
                K = row.numel()
                P = max(1, math.ceil(math.sqrt(K)))
                m = M_max if M_max is not None else max(1, math.ceil(multiplier * K / 2))
                if return_metadata:
                    result[i], meta = mpgbp(row, M_max=m, P=P,
                                            epsilon=epsilon, return_metadata=True)
                    _merge_meta(agg, meta)
                else:
                    result[i] = mpgbp(row, M_max=m, P=P, epsilon=epsilon)
        return (result, agg) if return_metadata else result

    elif mode == "L":
        flat = tensor.detach().reshape(-1)
        N = flat.numel()
        P = max(1, math.ceil(math.sqrt(N)))
        if M_max is not None:
            m = M_max
        else:
            if tensor.dim() == 1:
                K, R = N, 1
            else:
                K = tensor[0].numel()
                R = tensor.shape[0]
            m = max(1, math.ceil(multiplier * K * R / 4))
        if return_metadata:
            compressed, meta = mpgbp(flat, M_max=m, P=P,
                                     epsilon=epsilon, return_metadata=True)
            out = compressed.reshape(tensor.shape)
            return out, meta
        return mpgbp(flat, M_max=m, P=P, epsilon=epsilon).reshape(tensor.shape)

    else:
        raise ValueError(f"Unknown MPGBP mode: '{mode}'. Use 'S', 'R', 'L', or 'N' (N via apply_mpgbp).")


def apply_mpgbp(params: list, M_max: int = None,
                multiplier: float = 1.0, epsilon: float = 1e-6) -> list:
    """
    Per-network MPGBP ("N" mode): concatenate all parameter tensors into one
    vector, run mpgbp() once, then slice results back into original shapes.

    Args:
        params:     List of (tensor, shape) tuples — e.g. [(p.data.cpu(), p.shape) ...]
        M_max:      Maximum total SPTs for the entire network. If given, used
                    directly (backwards compatible). If None, computed from multiplier.
        multiplier: Budget multiplier. M_max = ceil(multiplier * N / (2 * L))
                    where N = total parameters, L = number of parameter tensors.
                    NOTE: formula approximate — flag for advisor review.
        epsilon:    Convergence threshold.

    Returns:
        List of compressed tensors, same shapes as input.
    """
    shapes = [t.shape for t, _ in params]
    flat   = torch.cat([t.detach().reshape(-1).float() for t, _ in params])
    N      = flat.numel()
    P      = max(1, math.ceil(math.sqrt(N)))
    if M_max is not None:
        m = M_max
    else:
        L = len(params)
        # NOTE: conservative budget formula — validate with advisor.
        m = max(1, math.ceil(multiplier * N / (2 * L)))
    compressed = mpgbp(flat, M_max=m, P=P, epsilon=epsilon)
    results = []
    offset  = 0
    for (orig, _), shape in zip(params, shapes):
        numel = orig.numel()
        results.append(compressed[offset:offset + numel].reshape(shape).to(orig.dtype))
        offset += numel
    return results


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
        # Simplified estimate: M_max * (ceil(log2(N)) + 1 + 8) per quantization unit
        # where N = size of the vector being quantized.
        M_max = kwargs.get("M_max", 16)
        mode  = kwargs.get("mode", "R")
        if mode == "S":
            # Each scalar is quantized independently: N=1, log2(1)=0
            # bits per element = M_max * (0 + 1 + 8) = M_max * 9
            return numel * M_max * 9
        elif mode == "R":
            total = 0
            if tensor.dim() == 1:
                N = numel
                total = M_max * (math.ceil(math.log2(max(N, 2))) + 1 + 8)
            else:
                for i in range(tensor.shape[0]):
                    N = tensor[i].numel()
                    total += M_max * (math.ceil(math.log2(max(N, 2))) + 1 + 8)
            return total
        elif mode in ("L", "N"):
            # One vector: entire tensor (L) or entire network (N, caller handles N)
            N = numel
            return M_max * (math.ceil(math.log2(max(N, 2))) + 1 + 8)
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
        M_max      = kwargs.get("M_max", None)
        multiplier = kwargs.get("multiplier", 1.0)
        mode       = kwargs.get("mode", "R")
        epsilon    = kwargs.get("epsilon", 1e-6)
        return (
            lambda t: compress_mpgbp(t, M_max=M_max, multiplier=multiplier,
                                     mode=mode, epsilon=epsilon),
            lambda t: estimate_bits(t, "mpgbp", M_max=M_max, mode=mode),
        )

    else:
        raise ValueError(f"Unknown compression method: {method}")


# =============================================================================
# Quick self-test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)
    test_tensor = torch.randn(4, 3)
    print(f"Original:\n{test_tensor}\n")

    for mode in ["S", "R", "L"]:
        result = compress_mpgbp(test_tensor, M_max=16, mode=mode)
        mse = torch.mean((test_tensor - result) ** 2).item()
        print(f"Mode {mode}: MSE = {mse:.6f}, shape = {result.shape}")
