#!/bin/bash

##NECESSARY JOB SPECIFICATIONS
##SBATCH --export=ALL               # use ALL for non-Intel MPI (foss) and NONE for Intel MPI
#SBATCH --job-name=TestDASH         #Set the job name to "TestDASH"
#SBATCH --time=2-00:10:00           #Set the wall clock limit to 0Day, 1hr, 30min and 0secs
#SBATCH --nodes=32 	     	        #Request 16 nodes
#SBATCH --ntasks-per-node=1         #Request 1 tasks/cores per node
#SBATCH --cpus-per-task=1	        #Request 8 cores/threads per task
#SBATCH --mem=4G                   #Request 32GB (32GB) per node 
#SBATCH --output=TestMEDRG.%j        #Send stdout/err to "TestRG.[jobID]"

##OPTIONAL JOB SPECIFICATIONS
#SBATCH --account=132704168404      #Set billing account to 132704168404 (primary) / 132704169032 (test)
##SBATCH --mail-type=ALL            #Send email on all job events
##SBATCH --mail-user=tdey@fsu.edu   #Send all emails to email_address 

#First Executable Line
module purge
module load Anaconda3/2022.05
module load icc/2018.3.222-GCC-7.3.0-2.30 impi/2018.3.222
module load mpi4py/3.0.1-Python-3.6.6
module load GCC/10.3.0
module load networkx/2.2-Python-3.6.6

module list

mpirun --version

obj=$1
alg=$2

declare -a seeds=("8" "14" "25" "35") # ("8" "14" "25" "35" "42") Please use any seed value as you prefer

for j in ${!seeds[@]};
do
    seed=${seeds[$j]}
    cmd="timeout 43280 mpirun --bind-to none python3 -W ignore ExpSet_Journal2.py ${obj} ${alg} 1 ${seed}"
    echo ${cmd}
    $cmd
    echo "${alg} - Data generated for seed = '${seed}'"
done