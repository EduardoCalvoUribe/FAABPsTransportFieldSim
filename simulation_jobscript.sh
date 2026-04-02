#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=18:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=112

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load numba/0.58.1-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a

#Execute a Python program located in $HOME
python $HOME/simulation_space/runPhaseSpace.py

#Copy output directory from scratch to home
cp -r . /home/lbuise/output_dir/    

