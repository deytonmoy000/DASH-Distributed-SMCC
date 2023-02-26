#!/bin/bash

algo=$2   
nThreads=$3
pyfile=$1
nNodesMin=$4
nNodesMax=$5

prefix="experiment_results_output_data/"
under="_"
suffix=".csv"

declare -a objs=("IFM2" "RVM2" "BA") #

for i in ${!objs[@]};
do
	nNodesMintmp=${nNodesMin}
	obj=${objs[$i]}
	while [[ ${nNodesMintmp} -le ${nNodesMax} ]] ; do
    	
		data=${nNodesMintmp}
		cmd="mpirun -np ${data} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} ${algo} ${nThreads}"
		echo $cmd
		# $cmd
		((nNodesMintmp *= 2))
	done
	echo "Exp3 Data generated for '${obj}'"
done

# Plot Fig 3 plots - Stored in plots/Fig4
cmd="bash plot_scripts/plot_Fig3.sh"
echo $cmd
$cmd