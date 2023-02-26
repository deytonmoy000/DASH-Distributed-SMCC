nNodes=$2   
nThreads=$3
pyfile=$1

prefix="experiment_results_output_data/"
under="_"
suffix=".csv"

declare -a objs=("IS" "IFM2" "RVM2" "BA")

for i in ${!objs[@]};
do
	obj=${objs[$i]}
	cmd1="timeout 86400 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} DASH ${nThreads}"
	cmd2="timeout 86400 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} PGBD ${nThreads}"
	
	echo $cmd1 # R-DASH
	$cmd1
	 
	echo $cmd2 # LS+PGB
	$cmd2
	
	echo "Data generated for '${data}'"
done

# Plot Fig 6 (bottom row - Distributed Data) - Stored in plots/Fig3Large
cmd="bash plot_scripts/plot_Fig6Large.sh"
echo $cmd
$cmd