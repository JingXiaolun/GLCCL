#!/bin/bash
#SBATCH --job-name=xclip_model_training
#SBTAHC --partition=debug  
#SBATCH -N 1 
#SBATCH -n 8
#SBTACH --ntasks-per-node=8
#SBATCH --mem-per-cpu=64g
#SBATCH --gres=gpu:4
#SBATCH -w node1
#SBATCH --output=../../../Slurm/%j.out
#SBATCH --error=../../../Slurm/%j.err
#SBATCH -t 12:00:00                            
#SBATCH --mail-type=end
#SBATCH --mail-user=jingxiaolun@sjtu.edu.cn

cd $SLURM_SUBMIT_DIR
sh main_offical.sh



