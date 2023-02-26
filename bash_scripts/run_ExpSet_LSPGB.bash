nNodes=$2   
nThreads=$3
pyfile=$1

prefix="experiment_results_output_data/Exp1"
under="_"
suffix=".csv"

# rm -f ${prefix}/*.csv # WILL REMOVE EXISTING RESULTS, OTHERWISE WILL START OVERWRITING THE EXISTING RESULTS

declare -a seeds=("42" "14" "25" "8" "35") # Please use any seed value as you prefer
declare -a objs=("IS" "IFM3" "RVM3" "BA") 
x=0
for j in ${!seeds[@]};
do
	
	for i in ${!objs[@]};
	do
		obj=${objs[$i]}
		seed=${seeds[$j]}
		
		cmd1="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} DASH ${nThreads} ${seed}"
		cmd2="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} PGBD ${nThreads} ${seed}"
		
		if [ "$seed" == "42" ]; then
			echo $cmd1 # R-DASH
			$cmd1
			
			echo $cmd2 # LS+PGB
			$cmd2

		else
			echo $cmd1 # R-DASH
			$cmd1
			
			echo $cmd2 # LS+PGB
			$cmd2

		fi

    done
    x=$((x+1))
	cmd0="mkdir -p experiment_results_output_data/Exp1/Rep${x}_New"
	echo $cmd0
	$cmd0
	
	cmd1="mv experiment_results_output_data/Exp1/*.csv experiment_results_output_data/Exp1/Rep${x}_New"
	echo $cmd1
	$cmd1
	echo "Data generated for seed = '${seed}'"
done

# Plot Fig 6 (upper row - Centralized Data) - Stored in plots/Fig3Small
cmd="bash plot_scripts/plot_Fig6Small.sh"
echo $cmd
$cmd
