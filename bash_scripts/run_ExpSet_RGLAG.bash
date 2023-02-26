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
		cmd3="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGGB ${nThreads} ${seed}"
		cmd9="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGLAG ${nThreads} ${seed}"
		
		if [ "$seed" == "42" ]; then
			echo $cmd1 # R-DASH
			$cmd1
			
			echo $cmd3 # RandGreedI
			$cmd3

			echo $cmd9 # RandGreedI+LAG
			$cmd9

		else
			echo $cmd1 # R-DASH
			$cmd1

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

# Plot Fig 1,4 plots - Stored in plots/Fig1
cmd="bash plot_scripts/plot_Fig1.sh "
echo $cmd
$cmd

# Plot Fig 6 (upper row - Centralized Data) - Stored in plots/Fig3Small
cmd="bash plot_scripts/plot_Fig3Small.sh"
echo $cmd
$cmd