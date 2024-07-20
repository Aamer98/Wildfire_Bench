#!/bin/bash
#SBATCH --mail-user=ar.aamer@gmail.com
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL
#SBATCH --job-name=WildfireSpreadTS_seed1
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=50000M
#SBATCH --time=3-00:00
#SBATCH --account=rrg-ebrahimi

nvidia-smi

module load StdEnv/2023  gcc/12.3 netcdf/4.9.2
source ~/my_env/climaX/bin/activate
wandb offline


echo "------------------------------------------<Move Data>---------------------------------------"
cd $SLURM_TMPDIR
mkdir data
cd data
cp -r /home/aamer98/projects/def-ebrahimi/aamer98/data/wildfirespreadTS_hdf5.zip .
unzip wildfirespreadTS_hdf5.zip
rm wildfirespreadTS_hdf5.zip


echo "---------------------------------------<Run the program>------------------------------------"
date +"%T"

cd $SLURM_TMPDIR
cp -r /home/aamer98/projects/def-ebrahimi/aamer98/repos/Wildfire_Bench .
cd Wildfire_Bench

python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --seed_everything=1 --trainer.devices=1 --data.root_dir=$SLURM_TMPDIR/data


echo "----------------------------------------<Move logs>------------------------------------"
date +"%T"

cp -r wandb /home/aamer98/projects/def-ebrahimi/aamer98/repos/Wildfire_Bench
cp -r logs /home/aamer98/projects/def-ebrahimi/aamer98/repos/Wildfire_Bench


echo "----------------------------------------<End of program>------------------------------------"
date +"%T"