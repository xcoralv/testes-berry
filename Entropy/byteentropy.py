from typing import Optional
import time
import os
import logging
import subprocess
import sys
import numpy as np
from collections import Counter
import math


from berry import log

try:
    import berry._subroutines.loadmeta as m
    import berry._subroutines.loaddata as d
except:
    pass



class WfcGenerator:
    def __init__(self, 
                 nk_points: Optional[int] = None , 
                 bands: Optional[int] = None,
                 npr: Optional[int]= None , 
                 logger_name: str = "genwfc", 
                 logger_level: int = logging.INFO, 
                 flush: bool = False
                ):
        
        if bands is not None and nk_points is None:
            raise ValueError("To generate a wavefunction for a single band, you must specify the k-point.")

        os.system("mkdir -p " + m.wfcdirectory)
        self.npr = npr if npr is not None else m.npr

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
       ''' (...)'''


    def byte_entropy_from_bytes(self, b: bytes) -> float:
            if len(b) == 0:
                return 0.0
            counts = Counter(b)
            total = len(b)
            H = 0.0
            for cnt in counts.values():
                p = cnt / total
                H -= p * math.log2(p)
            return H  # bits per byte (max 8)

    def byte_entropy_of_array(self, arr: np.ndarray) -> float:
        return self.byte_entropy_from_bytes(arr.tobytes())


    def analyze_psi_entropy(self, psi: np.ndarray):
        print("=== Byte-level entropy ===")
        raw_bytes_entropy = self.byte_entropy_of_array(psi)
        print(f"Entropy of psi.tobytes(): {raw_bytes_entropy:.3f} bits/byte (max=8)")
        predicted_cr = 8.0 / raw_bytes_entropy if raw_bytes_entropy > 0 else float('inf')
        print(f"Predicted maximum lossless CR ≈ {predicted_cr:.3f}x")
    
    def _wfck2r(self, nk_point: int, initial_band: int, number_of_bands: int):
        # Set the command to run
        shell_cmd = self._get_command(nk_point, initial_band, number_of_bands)

        # Runs the command
        output = subprocess.check_output(shell_cmd, shell=True)
        print('length output', len(output))
        # Converts fortran complex numbers to numpy format
        out1 = (output.decode("utf-8")
                .replace(")", "j")
                .replace(", -", "-")
                .replace(",  ", "+")
                .replace("(", "")
                )
        
        
        

        if m.noncolin:
            '''(...)'''
        else:
            # puts the wavefunctions into a numpy array
            psi = np.fromstring(out1, dtype=complex, sep="\n")

            # For each band, find the value of the wfc at the specific point rpoint (in real space)
            psi_rpoint = np.array([psi[int(m.rpoint) + m.nr * i] for i in range(number_of_bands)])

            # Calculate the phase at rpoint for all the bands
            deltaphase = np.arctan2(psi_rpoint.imag, psi_rpoint.real)

            # and the modulus of the wavefunction at the reference point rpoint (
            # will be used to verify if the wavefunction at rpoint is significantly different from zero)
            mod_rpoint = np.absolute(psi_rpoint)

            psifinal = []
            
            for i in range(number_of_bands):
                self.logger.debug(f"\t{nk_point:6d}  {(i + initial_band):4d}  {mod_rpoint[i]:12.8f}  {deltaphase[i]:12.8f}   {not mod_rpoint[i] < 1e-5}")
                
                # Subtract the reference phase for each point
                psifinal += list(psi[i * m.nr : (i + 1) * m.nr] * np.exp(-1j * deltaphase[i]))
                
            psifinal = np.array(psifinal)

            self.analyze_psi_entropy(psifinal)
            outfiles = map(lambda band: os.path.join(m.wfcdirectory, f"k0{nk_point}b0{band+initial_band}.wfc"), range(number_of_bands))
            
            for i, outfile in enumerate(outfiles):
                with open(outfile, "wb") as fich:
                    np.save(fich, psifinal[i * m.nr : (i + 1) * m.nr])


'''(...)'''
