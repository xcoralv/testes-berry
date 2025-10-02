import argparse
import numpy as np
import os

# ---------- Utilities ----------

def l2_rel_error(a, b):
    """Relative L2 norm error."""
    return np.linalg.norm(a - b) / np.linalg.norm(a)

def overlap_error(a, b):
    """Deviation from perfect overlap (normalized inner product)."""
    num = np.vdot(a, b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return 1 - abs(num) / denom if denom != 0 else np.nan

# ---------- Compression tricks ----------

def downcast_float(arr, dtype):
    return arr.astype(dtype)

def threshold_array(arr, thresh, rel=False):
    if rel:
        norm = np.max(np.abs(arr))
        thresh = thresh * norm
    out = arr.copy()
    out[np.abs(out) < thresh] = 0.0
    return out

def trunc_mantissa_float64_array(arr, keep_bits):
    """Keep only the top `keep_bits` mantissa bits in a float64 array."""
    #a = np.asarray(arr, dtype=np.float64).ravel()  # ensure pure float64
    a = np.ascontiguousarray(arr, dtype=np.float64)
    print(a.dtype)
    bits = a.view(np.uint64)
    #mantissa_mask = (1 << 52) - 1
    mantissa_mask = np.uint64((1 << 52) - 1)
    exponent_and_sign = bits & (~mantissa_mask)
    mantissa = bits & mantissa_mask
    shift = 52 - keep_bits
    mantissa = (mantissa >> shift) << shift
    new_bits = exponent_and_sign | mantissa
    return new_bits.view(np.float64).reshape(arr.shape)

# ---------- Experiment runner ----------

def run_experiments(psi, threshold=None, rel_threshold=None, truncate_bits=None):
    results = {}

    # 1. Downcasts
    results["float32_downcast"] = downcast_float(psi, np.complex64)
    results["float16_downcast"] = (psi.real.astype(np.float16) 
                                   + 1j * psi.imag.astype(np.float16))

    # 2. Thresholding
    if threshold is not None:
        results[f"abs_thresh_{threshold:.1e}"] = threshold_array(psi, threshold, rel=False)
    if rel_threshold is not None:
        results[f"rel_thresh_{rel_threshold:.1e}"] = threshold_array(psi, rel_threshold, rel=True)

    # 3. Mantissa truncation
    if truncate_bits:
        for kb in truncate_bits:
            real_trunc = trunc_mantissa_float64_array(psi.real, kb)
            imag_trunc = trunc_mantissa_float64_array(psi.imag, kb)
            results[f"trunc_{kb}bits"] = real_trunc + 1j * imag_trunc

    return results

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("psi_file", help="Path to input .wfc file (np.save format)")
    ap.add_argument("--out", default=None, help="Directory to save results")
    ap.add_argument("--threshold", type=float, default=None, help="Absolute threshold")
    ap.add_argument("--rel-threshold", type=float, default=None, help="Relative threshold")
    ap.add_argument("--truncate-bits", type=int, nargs="+", help="Mantissa bits to keep")
    args = ap.parse_args()

    psi = np.load(args.psi_file)  # assumes psifinal array saved directly
    print(f"Loaded {args.psi_file}, dtype={psi.dtype}, shape={psi.shape}")
    print(psi.real.dtype)  # should be float64

    results = run_experiments(
        psi,
        threshold=args.threshold,
        rel_threshold=args.rel_threshold,
        truncate_bits=args.truncate_bits,
    )

    # Report
    for name, arr in results.items():
        l2 = l2_rel_error(psi, arr)
        overlap_dev = overlap_error(psi, arr)
        raw_bytes = arr.nbytes
        print(f"{name:25s} | L2_rel: {l2:.3e} | overlap_dev: {overlap_dev:.3e} | raw: {raw_bytes:10d}")

        if args.out:
            os.makedirs(args.out, exist_ok=True)
            np.save(os.path.join(args.out, f"{name}.npy"), arr)

if __name__ == "__main__":
    main()
