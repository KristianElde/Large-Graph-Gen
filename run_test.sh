#!/bin/bash
#SBATCH --job-name=run_test
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --mem=256G
#SBATCH --time=32:00:00
#SBATCH --output=outputs/run_test_%A.out
#SBATCH --error=outputs/run_test_%A.err

module purge
module load 2024
module load GCC/13.3.0
module load Miniconda3/24.7.1-0
module load Mamba/24.9.0-0

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate large-graph-gen

# Ensure clean environment (isolate from system modules)
unset PYTHONPATH
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

# Run test
python finetune.py   --model GSAI-ML/LLaDA-8B-Instruct   --graph-tokenizer-type autograph   --pyg-dataset MUTAG   --max-graphs 16   --epochs 1   --train-device cpu   --torch-dtype float32