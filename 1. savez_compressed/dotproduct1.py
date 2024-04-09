from multiprocessing import Pool, Array
from typing import Tuple
import sys
import os
from time import time
import ctypes
import logging

import numpy as np

from berry import log

try:
    import berry._subroutines.loaddata as d
    import berry._subroutines.loadmeta as m
except:
    pass


def dot(nk: int, j: int, neighbor: int, jNeighbor: Tuple[np.ndarray]) -> None:
    start = time()

    dphase = d_phase[:, nk] * d_phase[:, neighbor].conj()

    if m.noncolin:  # Noncolinear case
        for band0 in range(m.nbnd):
            wfc00 = np.load(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}-0.wfc"))
            wfc01 = np.load(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}-1.wfc"))
            for band1 in range(m.nbnd):
                wfc10 = np.load(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}-0.wfc")).conj()
                wfc11 = np.load(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}-1.wfc")).conj()
                
                # not normalized dot product
                dpc[nk, j, band0, band1] = np.einsum("k,k,k->", dphase, wfc00, wfc10) + np.einsum("k,k,k->", dphase, wfc01, wfc11)
                dpc[neighbor, jNeighbor, band1, band0] = dpc[nk, j, band0, band1].conj()
                logger.debug(f"\t{nk}\t{band0}\t{j}\t{band1}\t",str(dpc[nk, j, band0, band1]))
        #print(sys.getsizeof(wfc10))
        #print(type(wfc10))
    else:  # Non-relativistic case
        #te = 0
        #c=0
        for band0 in range(m.nbnd):
            #start_time1 = time()
            wfc0 = np.load(os.path.join(m.wfcdirectory, f"k0{nk}b0{band0}.wfc"))
            wfc0 = wfc0['a']
            #end_time1 = time()
            #time_elapsed1 = end_time1 - start_time1
            #te += time_elapsed1
            #c+=1
            #print('time elapsed 1',time_elapsed1)
            for band1 in range(m.nbnd):
                #start_time2 = time()
                wfc1 = np.load(os.path.join(m.wfcdirectory, f"k0{neighbor}b0{band1}.wfc"))
                #print(type(wfc1))
                wfc1 = wfc1['a'].conj()
                #print(type(wfc1))
                #end_time2 = time()
                #time_elapsed2 = end_time2 - start_time2
                #te += time_elapsed2
                #c+=1
                #print('time elapsed 2',time_elapsed2)
                # not normalized dot product
                dpc[nk, j, band0, band1] = np.einsum("k,k,k->", dphase, wfc0, wfc1)
                dpc[neighbor, jNeighbor, band1, band0] = dpc[nk, j, band0, band1].conj()
                logger.debug(f"\t{nk}\t{band0}\t{j}\t{band1}\t",str(dpc[nk, j, band0, band1]))

    logger.debug(f"\tFinished of nk: {nk:>4}\tneighbor: {neighbor:>4}\tin: {(time() - start):>4.2f} seconds")


def get_point_neighbors(nk: int, j: int) -> None:
    """Generates the arguments for the pre_connection function."""
    neighbor = d.neighbors[nk, j]
    if neighbor != -1 and neighbor > nk:
        jNeighbor = np.where(d.neighbors[neighbor] == nk)

        return (nk, j, neighbor, jNeighbor)
    return None

def run_dot(npr: int = 1, logger_name: str = "dot", logger_level: logging = logging.INFO, flush: bool = False):
    global dpc, logger, d_phase
    logger = log(logger_name, "DOT PRODUCT", level=logger_level, flush=flush)

    if not 0 < npr <= os.cpu_count():
        raise ValueError(f"npr must be between 1 and {os.cpu_count()}")

    logger.header()

    ###########################################################################
    # 1. DEFINING THE CONSTANTS
    ########################################################################### 
    DPC_SIZE = m.nks * 2 * m.dimensions * m.nbnd * m.nbnd
    DPC_SHAPE = (m.nks, 2 * m.dimensions, m.nbnd, m.nbnd)

    ###########################################################################
    # 2. STDOUT THE PARAMETERS
    ########################################################################### 
    logger.info(f"\tUnique reference of run: {m.refname}")
    logger.info(f"\tNumber of processors to use: {npr}")
    logger.info(f"\tNumber of bands: {m.nbnd}")
    logger.info(f"\tTotal number of k-points: {m.nks}")
    logger.info(f"\tTotal number of points in real space: {m.nr}")
    logger.info(f"\tDirectory where the wfc are: {m.wfcdirectory}\n")

    ###########################################################################
    # 3. CREATE ALL THE ARRAYS
    ###########################################################################
    dpc_base = Array(ctypes.c_double, 2 * DPC_SIZE, lock=False)
    dpc = np.frombuffer(dpc_base, dtype=np.complex128).reshape(DPC_SHAPE)
    dp = np.zeros(DPC_SHAPE, dtype=np.float64)
    d_phase = np.load(os.path.join(m.workdir, os.path.join(m.data_dir, "phase.npy")))

    ###########################################################################
    # 4. CALCULATE
    ###########################################################################
    with Pool(npr) as pool:
        pre_connection_args = (
            args
            for nk in range(m.nks)
            for j in range(2 * m.dimensions)
            if (args := get_point_neighbors(nk, j)) is not None
        )
        
        pool.starmap(dot, pre_connection_args)
        
        
    dpc /= m.nr         # To normalize the dot product
    dp = np.abs(dpc)    # Calculate the modulus of the dot product

    ###########################################################################
    # 5. SAVE OUTPUT
    ###########################################################################
    np.save(os.path.join(m.data_dir, "dpc.npy"), dpc)
    np.save(os.path.join(m.data_dir, "dp.npy"), dp)
    logger.info(f"\n\tDot products saved to file dpc.npy")
    logger.info(f"\tDot products modulus saved to file dp.npy")

    ###########################################################################
    # Finished
    ###########################################################################
    logger.footer()

if __name__ == "__main__":
    #run_dot(log("dotproduct", "DOT PRODUCT", "version"), 20)
    start_timef = time()
    run_dot()
    end_timef = time()
    print('total dot time:', end_timef-start_timef)