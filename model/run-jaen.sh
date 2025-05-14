#!/bin/bash
#SBATCH --account=project_2014146
#SBATCH --output=output_%j.txt
#SBATCH --partition=gputest
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=32G
#SBATCH --time=15
#SBATCH --gres=gpu:v100:4,nvme:10

module purge
module load pytorch

set -x

tar xf /scratch/project_2014146/janika-kinnunen/jpchar-data.tar.gz -C $LOCAL_SCRATCH

srun python3 jaen_cnn.py --version 1 --data_path=$LOCAL_SCRATCH/jpchar-data --label_file=/scratch/project_201416/janika-kinnunen/label.txt

seff $SLURM_JOBID