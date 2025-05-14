from typing import Optional

import re
from functools import partial
from multiprocessing import Pool, Array
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
import threading
import concurrent.futures
import os
import logging
import subprocess
import time
import numpy as np
from collections import ChainMap

from berry import log

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
    import berry._subroutines.parallel_save as p
except:
    pass

class WfcGenerator:
    def __init__(self, 
                 nk_points: Optional[int] = None , 
                 bands: Optional[int] = None, 
                 logger_name: str = "genwfc", 
                 logger_level: int = logging.INFO, 
                 flush: bool = False
                ):

        if bands is not None and nk_points is None:
            raise ValueError("To generate a wavefunction for a single band, you must specify the k-point.")

        os.system("mkdir -p " + m.wfcdirectory)
        self.outfile = os.path.join(m.wfcdirectory, "mainfile.npy")
        if nk_points is None:
            self.nk_points = range(m.nks)
        elif  bands is None:
            self.nk_points = nk_points
            self.bands = range(m.nbnd)
        else:
            self.nk_points = nk_points
            self.bands = bands
        self.ref_name = m.refname
        self.logger = log(logger_name, "GENERATE WAVE FUNCTIONS", level=logger_level, flush=flush)

    def run(self):
        # prints header on the log file
        self.logger.header()

        # Logs the parameters for the run
        self._log_run_params()

        # Sets the program used for converting wavefunctions to the real space
        if m.noncolin:
            self.k2r_program = "wfck2rFR.x"
            self.logger.info("\tNoncolinear calculation, will use wfck2rFR.x")
        else:
            self.k2r_program = "wfck2r.x"
            self.logger.info("\tNonrelativistic calculation, will use wfck2r.x")

        # Set which k-points and bands will use (for debuging)
        if isinstance(self.nk_points, range):
                self.logger.info(f"\tWill run for k-point {self.nk_points} and all bands")
                self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")

                self.logger.info(f"\tCalculating wfc for k-point {self.nk_points}")
                # Set the command to run
                shell_cmd = self._get_command(m.nks, 0, m.nbnd)

                # Runs the command
                output = subprocess.check_output(shell_cmd, shell=True)
                print('output:', len(output))
                # Converts fortran complex numbers to numpy format
               # out1 = (output.decode("utf-8")
                #            .replace(")", "j")
                 #           .replace(", -", "-")
                  #          .replace(",  ", "+")
                   #         .replace("(", "")
                    #        )
                
                # Define the pattern to match all cases
                pattern = r"\(|\)|, -|,  "  

                # Apply re.sub with clean_output function
                out1 = re.sub(pattern, self.clean_output, output.decode("utf-8"))
                # puts the wavefunctions into a numpy array
                #psitotal = np.fromstring(out1, dtype=complex, sep="\n")
                psitotal = self.process_large_string_parallel(out1, m.npr)
                print('psitotal:', len(psitotal))

                k_slices = {nk: psitotal[m.nr * m.nbnd * nk : m.nr * m.nbnd * (nk + 1)] for nk in self.nk_points}

                # **Parallel Processing for Each k-point**
                with Pool(processes=1) as pool:
                    results = pool.starmap(self._wfck2r, [(nk, k_slices[nk], m.nbnd) for nk in self.nk_points])

                # Combine results into a single array
                if any(result is None for result in results):
                    raise ValueError("Some tasks failed. Check your _wfck2r function.")

                # Combine all dictionaries into one
                all_portion_dict = dict(ChainMap(*results))
                # Now save all data for all nk_points into the file
                with open(self.outfile, "wb") as fich:
                    np.savez(fich, **all_portion_dict)

        else: 
            if isinstance(self.bands, range):
                self.logger.info(f"\tWill run for k-point {self.nk_points} and all bands")
                self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")

                self.logger.info(f"\tCalculating wfc for k-point {self.nk_points}")
                self._wfck2r(self.nk_points, 0, m.nbnd)

            else:
                self.logger.info(f"\tWill run just for k-point {self.nk_points} and band {self.bands}.\n")
                self._wfck2r(self.nk_points, self.bands, 1)

        self.logger.info("\n\tRemoving temporary file 'tmp'")

        os.system(f"rm {os.getcwd()}/tmp")
        self.logger.info(f"\tRemoving quantum expresso output file '{m.wfck2r}'")
        os.system(f"rm {os.path.join(os.getcwd(),m.wfck2r)}")


        self.logger.footer()
    
    def clean_output(self, match):
        if match.group() == ")":
            return "j"
        elif match.group() == "(":
            return ""
        elif match.group() == ", -":
            return "-"
        elif match.group() == ",  ":
            return "+"


    # Function to parse a chunk of the string and convert it to complex values
    def parse_chunk(self, chunk):
        #return np.fromiter(chunk.split('\n'), dtype=complex)
        #return np.fromstring(chunk, dtype=complex, sep="\n")
        return np.fromiter((num for num in chunk.split('\n') if num), dtype=complex)

    # Function to process the string in parallel
    def process_large_string_parallel(self, large_string, num_workers):
        # Split the large string into chunks
        #chunk_sizes = [len(line) for line in large_string.split('\n')]
        #chunk_sizes = chunk_sizes[:-1]
        #print(chunk_sizes[0])
        #print(len(set(chunk_sizes)) == 1)
        chunk_size = len(large_string) // num_workers
        chunks = [large_string[i:i+chunk_size] for i in range(0, len(large_string), chunk_size)]
        for i in range(0, len(large_string), chunk_size):
            chunk = large_string[i:i+chunk_size]
            print('chunk:', len(chunk))

        # Create a pool of workers for parallel processing
        with multiprocessing.Pool(processes=num_workers) as pool:
            # Apply the parse_chunk function to each chunk
            results = pool.map(self.parse_chunk, chunks)

        # Combine the results into a single numpy array
        return np.concatenate(results)

    def _log_run_params(self):
        self.logger.info(f"\tUnique reference of run: {self.ref_name}")
        self.logger.info(f"\tWavefunctions will be saved in directory {m.wfcdirectory}")
        self.logger.info(f"\tDFT files are in directory {m.dftdirectory}")
        self.logger.info(f"\tThis program will run in {m.npr} processors\n")

        self.logger.info(f"\tTotal number of k-points: {m.nks}")
        self.logger.info(f"\tNumber of r-points in each direction: {m.nr1} {m.nr2} {m.nr3}")
        self.logger.info(f"\tTotal number of points in real space: {m.nr}")
        self.logger.info(f"\tNumber of bands: {m.nbnd}\n")

        self.logger.info(f"\tPoint choosen for sincronizing phases:  {m.rpoint}\n")

    def _wfck2r(self, nk_point: int, psi: np.ndarray, number_of_bands: int):

        if m.noncolin:
            x=1
        else:

            print('psi',len(psi))

            psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(m.nbnd)])

            # Calculate the phase at rpoint for all the bands
            deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

            # and the modulus of the wavefunction at the reference point rpoint (
            # will be used to verify if the wavefunction at rpoint is significantly different from zero)
            mod_rpoint = np.absolute(psi_rpoint)

            nk_dict = {}       
            for i in range(m.nbnd):
                self.logger.debug(f"\t{nk_point:6d}  {i:4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")

                # Subtract the reference phase for each point
                psi_band = psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[i])

                # Define the key name for the portion (e.g., 'portion_0', 'portion_1', etc.)
                key_name = f'k0{nk_point}band0{i}'

                # Add the portion to the dictionary with the unique key name
                nk_dict[key_name] = psi_band
            return nk_dict

    def _get_command(self, nk_points: int, initial_band: int, number_of_bands: int):
        #with self.lock:
            mpi = "" if m.npr == 1 else f"mpirun -np {m.npr} "
            command =f"&inputpp prefix = '{m.prefix}',\
                            outdir = '{m.outdir}',\
                            first_k = {1},\
                            last_k = {m.nks},\
                            first_band = {initial_band + 1},\
                            last_band = {initial_band + number_of_bands},\
                            loctave = .true., /"
            if m.noncolin:
                return f'echo "{command}" | {mpi} wfck2rFR.x > tmp; tail -{m.nr * number_of_bands*2} {m.wfck2r}'
            else:
                return f'echo "{command}" | {mpi} wfck2r.x > tmp; tail -{m.nr * number_of_bands * m.nks} {m.wfck2r}'
