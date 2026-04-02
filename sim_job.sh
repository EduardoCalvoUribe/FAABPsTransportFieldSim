#!/bin/bash
#Set job requirements
#SBATCH --job-name=FAABPsPolarity
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --time=1:00:00
#SBATCH --partition=rome
#SBATCH --cpus-per-task=1

#Loading modules
module load 2023
module load Python/3.11.3-GCCcore-12.3.0
module load numba/0.58.1-foss-2023a
module load SciPy-bundle/2023.07-gfbf-2023a
module load matplotlib/3.7.2-gfbf-2023a

#Execute simulation
cd $HOME/FAABPsTransportFieldSim
python main.py

#Copy output directories from scratch to home # cd $TMPDIR
cp -r visualizations/ $HOME/FAABPsTransportFieldSim/visualizations/
cp -r data/ $HOME/FAABPsTransportFieldSim/data/

