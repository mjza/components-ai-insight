#!/bin/bash
#SBATCH --job-name=BERTSE
#SBATCH --output=/work/barcomb_lab/Mahdi/components-ai-insight/logs/job_output_%j.log
#SBATCH --error=/work/barcomb_lab/Mahdi/components-ai-insight/logs/job_error_%j.log
#SBATCH --time=7-00:00:00  # Max time: 7 days
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  # Increase CPUs for better performance
#SBATCH --mem=16G  # Adjust memory as needed
#SBATCH --partition=gpu-v100
#SBATCH --gres=gpu:1  # Request 1 GPU

####### Load necessary modules ###############
module purge
module load python/3.12.5
module load cuda/11.8  # Adjust CUDA version based on availability

# Assign SLURM GPU to CUDA environment
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Prevents TensorFlow from allocating all GPU memory

####### Set up virtual environment ###############
VENV_DIR="/work/barcomb_lab/Mahdi/components-ai-insight/senv"
if [ ! -d "$VENV_DIR" ]; then
    python -m venv "$VENV_DIR"
    source "$VENV_DIR/bin/activate"
    pip install --upgrade pip
    pip install -r /work/barcomb_lab/Mahdi/components-ai-insight/requirements.txt
else
    source "$VENV_DIR/bin/activate"
fi

####### Cleanup Old Logs #########################
LOG_DIR="/work/barcomb_lab/Mahdi/components-ai-insight/logs"

# Delete logs older than 7 days
find "$LOG_DIR" -type f -name "job_*.log" -mtime +7 -exec rm {} \;
find "$LOG_DIR" -type f -name "job_error_*.log" -mtime +7 -exec rm {} \;

echo "✅ Old logs cleaned up successfully!"

####### Run the script with GPU #########################
python /work/barcomb_lab/Mahdi/components-ai-insight/09_update_bert_se_similarity.py
