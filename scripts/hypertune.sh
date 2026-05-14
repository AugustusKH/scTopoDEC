#!/bin/bash
#SBATCH --job-name=hypertune          # Job name
#SBATCH --output=output.log           # Stdout log
#SBATCH --error=error.log             # Stderr log
#SBATCH --time=2-00:00:00             # D-HH:MM:SS
#SBATCH --ntasks=1                    # Number of tasks
#SBATCH --cpus-per-task=8             # CPU cores per task
#SBATCH --mem=64G                     # Memory per node
#SBATCH --partition=gpu               # Partition name
#SBATCH --gres=gpu:1


# Load necessary HPC modules
module load gcc/12.1.0

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh 
conda activate sctopodec

# Point Linux to the CUDA libraries
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
for d in $CONDA_PREFIX/lib/python3.11/site-packages/nvidia/*/lib; do
  export LD_LIBRARY_PATH=$d:$LD_LIBRARY_PATH
done
export XLA_FLAGS=--xla_gpu_cuda_data_dir=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvcc

python -c "import tensorflow as tf; print('Num GPUs:', len(tf.config.list_physical_devices('GPU')))"

# Set Stability Variables
export TF_ENABLE_ONEDNN_OPTS=0       # Prevents NaN drift
export OMP_NUM_THREADS=8             # Matches cpus-per-task
export MKL_NUM_THREADS=8
export TF_NUM_INTRA_OP_THREADS=8
export TF_CPP_MIN_LOG_LEVEL=2        # Reduces log bloat in tune_out_%j.log

# Run the Optuna tuning script
# Adjust the input path to where your file is located
ulimit -s unlimited
export TF_FORCE_GPU_ALLOW_GROWTH=true
python run_hypertune.py
