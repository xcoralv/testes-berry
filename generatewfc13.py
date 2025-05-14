from typing import Optional

from functools import partial
from multiprocessing import Pool, Array
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from threading import Thread
import threading
import concurrent.futures
import os
import logging
import subprocess
import time
import numpy as np

#import numba
#print(numba.__version__)
#from numba import njit, prange
#import numpy as np
#from numba.openmp import openmp_context as openmp
#from numba.openmp import omp_get_thread_num, omp_get_num_threads
#MaxTHREADS = 32
#@njit

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
        #self.lock = threading.Lock()
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
            self.logger.info("\n\tWill run for all k-points and bands")
            self.logger.info(f"\tThere are {m.nks} k-points and {m.nbnd} bands.\n")
            #print('----aqui1')
            #for nk in self.nk_points:
            #self.logger.info(f"\tCalculating wfc for k-point {nk}")
            self._wfck2r(m.nks, 0, m.nbnd)
            #args = [nk for nk in self.nk_points]
            #print(args)
            #with Pool(processes=1) as pool:
             #   pool.map(self._parallel_wfck2r, args)

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

    #@njit(parallel=True)
    def _wfck2r(self, nk_points: int, initial_band: int, number_of_bands: int):
        # Set the command to run
        shell_cmd = self._get_command(nk_points, initial_band, number_of_bands)
        #print(shell_cmd)
        #print('1----------------------------------------------------------------------',nk_point)
            #print('1----------------------------------------------------------------------------------------------')
        # Runs the command
        #try:
        output = subprocess.check_output(shell_cmd, shell=True)
        print('len output:', len(output))
                #print(type(output))
        #except subprocess.CalledProcessError as e:
         #       print(f"Command failed with return code {e.returncode}")
        #print('len output', len(output))
        #print(output)
        out1 = (output.decode("utf-8")
                            .replace(")", "j")
                            .replace(", -", "-")
                            .replace(",  ", "+")
                            .replace("(", "")
                            )
        #print('out1 type', type(out1))
        print('len out1:', len(out1))
        #print('3----------------------------------------------------------------------',nk_point)
        if m.noncolin:
            x=1
            '''# puts the wavefunctions into a numpy array
            psi = np.fromstring(out1, dtype=complex, sep="\n")

            # For each band, find the value of the wfc at the specific point rpoint (in real space)
            psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(0,2*number_of_bands,2)])

            # Calculate the phase at rpoint for all the bands
            deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

            # and the modulus of the wavefunction at the reference point rpoint (
            # will be used to verify if the wavefunction at rpoint is significantly different from zero)
            mod_rpoint = np.absolute(psi_rpoint)

            psifinal0, psifinal1 = [], []

            for i in range(0,2*number_of_bands,2):
                self.logger.debug("\t{nk_point:6d}  {(int(i/2) + initial_band):4d}  {mod_rpoint[int(i/2)]:12.8f}  {deltaphase[int(i/2)]:12.8f}   {not mod_rpoint[int(i/2)] < 1e-5}")

                # Subtract the reference phase for each point
                psifinal0 += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))                # first part of spinor, all bands
                psifinal1 += list(psi[m.nr + i * m.nr : m.nr + (i + 1) * m.nr] * np.exp(-1j * deltaphase[int(i/2)]))  # second part of spinor, all bands

            outfiles0 = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-0.wfc"), range(number_of_bands))
            outfiles1 = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}-1.wfc"), range(number_of_bands))

            #for i, outfile in enumerate(outfiles0):
             #   with open(outfile, "wb") as fich:
              #      np.save(fich, psifinal0[i * m.nr : (i + 1) * m.nr])
            #for i, outfile in enumerate(outfiles1):
             #   with open(outfile, "wb") as fich:
              #      np.save(fich, psifinal1[i * m.nr : (i + 1) * m.nr])'''

        else:

            # puts the wavefunctions into a numpy array
            #print('4----------------------------------------------------------------------',nk_point)
            psi_total = np.fromstring(out1, dtype=complex, sep="\n")          #as primeiras são vazias ?
            print('psi total:', len(psi_total))
            k_slices = {nk: psi_total[m.nr * m.nbnd * nk : m.nr * m.nbnd * (nk + 1)] for nk in self.nk_points}
           # args = [nk for nk in self.nk_points]
           # partial_aux=partial(self.aux, psi_total)
            with Pool(processes=m.npr) as pool:
               # pool.map(partial_aux, args)
               results = pool.starmap(self._wfck2r, [(nk, k_slices[nk], m.nbnd) for nk in self.nk_points])
           # self.aux(psi_total, nk_points)
           # map(lambda nk: partial_aux, self.nk_points)
            
            '''for i in range(m.nks):
                psi = psi_total[m.nr * m.nbnd * i : m.nr * m.nbnd * (i+1)]
                print('psi',len(psi))
            #print('psi', psi[1])
            #print(len(psi), nk_point)
            #print('5----------------------------------------------------------------------',nk_point)
         #   print('4----------------------------------------------------------------------------------------------')
            # For each band, find the value of the wfc at the specific point rpoint (in real space)
            # Initialize psi_rpoint as a NumPy array of the appropriate size and data type
                psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(number_of_bands)])

             #   print(a)
            #print('length psi_rpoint:', len(psi_rpoint))
          #  print('5----------------------------------------------------------------------------------------------')
            # Calculate the phase at rpoint for all the bands
                deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

                # and the modulus of the wavefunction at the reference point rpoint (
                # will be used to verify if the wavefunction at rpoint is significantly different from zero)
                mod_rpoint = np.absolute(psi_rpoint)

                psifinal = []
                #psifinal = p.main(psi, deltaphase, m.nbnd)
                for i in range(number_of_bands):
                    self.logger.debug("\t{nk_point:6d}  {(i + initial_band):4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")

                    # Subtract the reference phase for each point
                    psifinal += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[i]))
                #print('length psifinal',len(psifinal))
                psifinal = np.array(psifinal)
                #print(len(psifinal))
                outfile = os.path.join(m.wfcdirectory, "k0{nk_point}.npy") #savez doesn't like to save with extensions other than .npz
                #print(os.getpid(), nk_point)
                with open(outfile, "wb") as fich:
                    np.save(fich, psifinal)'''


    def aux(self, psi_total, nk_point):
        psi = psi_total[m.nr * m.nbnd * nk_point : m.nr * m.nbnd * (nk_point+1)]
        print('len psi:',len(psi))
        print('nk_point:', nk_point)

# puts the wavefunctions into a numpy array
            # For each band, find the value of the wfc at the specific point rpoint (in real space)
        psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(m.nbnd)])

            # Calculate the phase at rpoint for all the bands
        deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

            # and the modulus of the wavefunction at the reference point rpoint (
            # will be used to verify if the wavefunction at rpoint is significantly different from zero)
        mod_rpoint = np.absolute(psi_rpoint)

        psifinal = []
            
        for i in range(m.nbnd):
            self.logger.debug(f"\t{nk_point:6d}  {i:4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")
                
            # Subtract the reference phase for each point
            psifinal += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[i]))
                
        psifinal = np.array(psifinal)

        outfile = os.path.join(m.wfcdirectory, f"k0{nk_point}.wfc")
            
        with open(outfile, "wb") as fich:
            np.save(fich, psifinal) 


   
                
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
