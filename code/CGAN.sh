#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=10G 
#SBATCH -p short
#SBATCH -o CGAN.out
#SBATCH --gres=gpu:1
python CGAN.py
echo Script is Complete!