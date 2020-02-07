#!/bin/bash
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=10G 
#SBATCH -p short
#SBATCH -o GAN.out
#SBATCH --gres=gpu:1
python GAN.py
echo Script is Complete!