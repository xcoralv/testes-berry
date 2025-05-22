# Como correr o Berry:
  "To run, first one has to create a working directory where the results will be saved, and
inside it create a directory called dft.
Inside dft should go the pseudopotential files for the atomic species of the dft calculation
and the file scf.in with the details of the scf run. You can use another name for the file,
but then you have to add a line in the input file of the script preprocess to change the
default (see section Preprocessing).
This scf.in file has to be a Quantum Espresso scf run file; this is the only one
implemented so far.
Create an input file (as described in chapter Workflow, section Preprocessing) in the
working directory."

  Ficheiros de input e dft na pasta "Inputs"

# Testes Berry
#### Métodos de compressão:
  1. savez_compressed
  2. gzip
  3. bz2
  4. lzma
#### Bandas agregadas

5. save
6. Savez – using dictionary

6.1 Memmap

8. multiprocessing

9. multithreading

9.1 multithreading + parallel construction of psi
#### Bandas e k-points num só ficheiro
  7. save


