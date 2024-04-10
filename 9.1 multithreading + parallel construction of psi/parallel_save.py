"""Write a NumPy array in parallel from multiple CPUs/processes, using shared memory."""

from contextlib import closing
import multiprocessing as mp
import os

import numpy as np

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
except:
    pass


import numpy as np
import multiprocessing as mp


def parallel_function(band_number, psi, deltaphase, psifinal):
    psifinal[band_number] = psi[band_number * m.nr : (band_number + 1) * m.nr] * np.exp(-1j * deltaphase[band_number])

def main(psi, deltaphase, number_of_bands):
    # Initialize shared array for psifinal
    shared_psifinal = mp.RawArray('d', number_of_bands * m.nr * 2)  # Double precision

    # Assign NumPy array views to the shared array
    psifinal_shared = np.frombuffer(shared_psifinal, dtype=np.complex128).reshape(number_of_bands * m.nr)

    # Set up multiprocessing Pool
    with closing(mp.Pool(processes=mp.cpu_count())) as pool:
        # Map parallel function to each band
        pool.starmap(parallel_function, [(i, psi, deltaphase, psifinal_shared) for i in range(number_of_bands)])

    # Convert psifinal_shared to a 1D array (if needed)
    psifinal = psifinal_shared.flatten()

    return psifinal


