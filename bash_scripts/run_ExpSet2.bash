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
	cmd3="timeout 86400 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGGB ${nThreads}"
	cmd4="timeout 86400 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} BCG ${nThreads}"
	cmd5="timeout 86400 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGLAG ${nThreads}"
	
	echo $cmd1 # R-DASH
	$cmd1
	 
	echo $cmd2 # LS+PGB
	$cmd2

	echo $cmd3 # RandGreedI
	$cmd3
	
	echo $cmd4 # BicCriteriaGreedy
	$cmd4

	# echo $cmd5
	# $cmd5
	
	echo "Data generated for '${data}'"
done

# Plot Fig 2,5 plots - Stored in plots/Fig2
cmd="bash plot_scripts/plot_Fig2.sh "
echo $cmd
$cmd