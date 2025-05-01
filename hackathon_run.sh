#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=256G
#SBATCH -o slurm.%j.out
#SBATCH -e slurm.%j.err
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:3
#SBATCH --partition=a100_full
#SBATCH --nodelist=agpu002

#SBATCH --mail-type=ALL 
#SBATCH --mail-user=YOURMAIL

# ADD DEBUGGING ECHO COMMANDS HERE
echo "Starting job at $(date)" 
echo "Working directory: $(pwd)"

module load python/3.11.7

source .venv/bin/activate

cd hyperspectralricekrispies+

srun python3 train.py


echo "Job completed at $(date)"