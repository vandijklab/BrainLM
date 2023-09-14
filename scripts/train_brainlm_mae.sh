#!/bin/bash
#SBATCH --job-name=brainlm                         # Job name
#SBATCH --output log_brainlm_%J.log                # Output log file
#SBATCH --mail-type=ALL                            # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=<USER>@gmail.com               # Email address to send status updates to
#SBATCH --partition pi_dijk                        # Train on private partition
#SBATCH --requeue
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1                        # Run on a single CPU
#SBATCH --gres=gpu:rtx3090:1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32gb                                 # Job memory request
#SBATCH --time=2-00:00:00                          # Time limit hrs:min:sec
date;hostname;pwd

module load miniconda
conda activate brainlm
cd /home/<USER>/projects/BrainLM/


# Training from scratch
python train.py \
    --output_dir training-runs/2023-05-01-22_00_00 \
    --train_dataset_path /home/<USER>/palmer_scratch/datasets/UKBioBank1000_Arrow/train_ukbiobank1000 \
    --val_dataset_path /home/<USER>/palmer_scratch/datasets/UKBioBank1000_Arrow/val_ukbiobank1000 \
    --coords_dataset_path /home/<USER>/palmer_scratch/datasets/UKBioBank1000_Arrow/Brain_Region_Coordinates  \
    --moving_window_len 20 \
    --num_last_timepoints_masked 4 \
    --hidden_size 128 \
    --num_hidden_layers 4 \
    --num_attention_heads 4 \
    --intermediate_size 512 \
    --decoder_hidden_size 128 \
    --decoder_num_hidden_layers 2 \
    --decoder_num_attention_heads 4 \
    --decoder_intermediate_size 512 \
    --attention_probs_dropout_prob 0.1 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 32 \
    --gradient_accumulation_steps 8 \
    --save_total_limit 50 \
    --dataloader_num_workers 3 \
    --dataloader_pin_memory True

