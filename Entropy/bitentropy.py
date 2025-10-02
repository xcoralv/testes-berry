# Before this file, a generatewfc version that produced five different files for each k-point/band combination was ran. Only k = 3 was considered for simplicity.
# Used files: "k03b00-1.wfc", "k03b00-2.wfc", "k03b00-3.wfc", "k03b00-4.wfc", "k03b00-5.wfc"
#             "k03b01-1.wfc", "k03b01-2.wfc", "k03b01-3.wfc", "k03b01-4.wfc", "k03b01-5.wfc"
#                                                   (...)
#             "k03b09-1.wfc", "k03b09-2.wfc", "k03b09-3.wfc", "k03b09-4.wfc", "k03b09-5.wfc"

import numpy as np
import os
import csv
import sys

# ---------- helpers ----------
def load_wfc_files(file_list):
    """Load multiple QE wavefunction arrays (numpy .wfc saved via np.save)."""
    psi_list = []
    for f in file_list:
        psi = np.load(f)
        psi_list.append(psi)
    return psi_list

def mantissa_bits(arr):
    """Return mantissa bits of a float64 array as uint64."""
    a = arr.astype(np.float64).ravel()
    bits = a.view(np.uint64)
    mantissa_mask = (1 << 52) - 1
    mantissas = bits & mantissa_mask
    return mantissas

def bit_stability_across_runs(psi_list):
    """
    Compute per-mantissa-bit stability across multiple runs.
    psi_list: list of np.ndarray arrays (same shape)
    Returns: dict with 'real' and 'imag' keys, each array of shape (52,)
             giving fraction of numbers unstable per bit.
    """
    shape = psi_list[0].shape
    N = np.prod(shape)
    results = {}

    is_complex = np.iscomplexobj(psi_list[0])
    parts = ['real', 'imag'] if is_complex else ['real']

    for part in parts:
        mantissa_arrs = []
        for psi in psi_list:
            x = psi.real if part == 'real' else psi.imag
            mantissa_arrs.append(mantissa_bits(x))
        mantissa_arrs = np.array(mantissa_arrs)  # shape (n_runs, N)

        nbits = 52
        unstable_fraction = np.zeros(nbits)
        for b in range(nbits):
            bitvals = ((mantissa_arrs >> b) & 1)  # shape (n_runs, N)
            unstable_per_number = (bitvals.max(axis=0) != bitvals.min(axis=0))
            unstable_fraction[b] = unstable_per_number.sum() / N
        results[part] = unstable_fraction
    return results

def analyze_noise_across_runs(file_list, k, b, writer=None):
    psi_list = load_wfc_files(file_list)
    unstable = bit_stability_across_runs(psi_list)
    
    row = {"k": k, "band": b}
    for part, uf in unstable.items():
        enob = np.sum(uf < 0.5)
        row[f"ENOB_{part}"] = enob
        print(f"\n=== Mantissa bit noise fraction ({part}) for k={k}, band={b} ===")
        print("Bit (MSB->LSB): Fraction unstable")
        for i in range(52):
            print(f"{i:2d} : {uf[51-i]:.3f}")
        print(f"Estimated ENOB ({part}) = {enob} bits")
    if writer is not None:
        writer.writerow(row)

# ---------- main ----------
if __name__ == "__main__":
    # Usage:
    # python analyze_enob.py <max_band> <n_runs> [initial_band]
    #
    # Example:
    # python analyze_enob.py 10 5 1
    # → analyze bands 1..10, 5 runs per band, starting at band 1
    
    if len(sys.argv) < 3:
        print("Usage: python analyze_enob.py <max_band> <n_runs> [initial_band]")
        sys.exit(1)

    max_band = int(sys.argv[1])
    n_runs = int(sys.argv[2])
    initial_band = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    base_dir = os.path.expanduser("~/InSe14/data/wfc")
    k = 3  # fixed k-point

    with open("enob_summary.csv", "w", newline="") as f:
        fieldnames = ["k", "band", "ENOB_real", "ENOB_imag"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for b in range(initial_band, max_band + 1):
            files = [os.path.join(base_dir, f"k0{k}b0{b}-{r}.wfc")
                     for r in range(0, n_runs)]

            files = [f for f in files if os.path.exists(f)]
            print(files)
            if len(files) == n_runs:
                print(f"\n=== Analyzing k-point {k}, band {b} ===")
                analyze_noise_across_runs(files, k, b, writer=writer)
            else:
                print(f"Skipping k={k}, band={b}: missing files")



# /usr/bin/time -f "Elapsed (wall clock) time: %e\nUser CPU time: %U\nSystem CPU time: %S\nCPU usage: %P\nMax memory: %M KB" berry wfcgen -np 20 0
# /usr/bin/time -f "Time: %E\n" python3.8 /home/carolfsg/berry-Version-2.0/berry/Entropy.py 9 5 0
