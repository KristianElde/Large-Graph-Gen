#!/bin/bash
#SBATCH --job-name=comm20_hyp_vqvae_install
#SBATCH --partition=gpu_a100
#SBATCH --gpus=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=9
#SBATCH --time=32:00:00
#SBATCH --output=outputs/comm20_vqvae_install_%A.out
#SBATCH --error=outputs/comm20_vqvae_install_%A.err

# 1. Environment
module purge
module load 2024
module load Miniconda3/24.7.1-0
module load Mamba/24.9.0-0

# 2. Create or update conda environment
mamba env remove -n large-graph-gen --yes

echo "Creating conda environment from environment.yaml..."
mamba env create -f environment.yaml --yes

# 3. Activate environment
echo "Activating environment..."
source activate large-graph-gen

# 4. Ensure clean environment (isolate from system modules)
unset PYTHONPATH
unset LD_LIBRARY_PATH

# 5. Verify installation
echo "Verifying installation..."
python --version
pip list
torch_python="import torch; import pandas; print(f'PyTorch version: {torch.__version__}'); print(f'Pandas version: {pandas.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "$torch_python"

# 6. Test torch-geometric import (main issue point)
echo "Testing torch-geometric import..."
python -c "from torch_geometric.datasets import TUDataset; print('torch-geometric import successful!')"

# 7. Installation complete
echo "Installation complete!"
