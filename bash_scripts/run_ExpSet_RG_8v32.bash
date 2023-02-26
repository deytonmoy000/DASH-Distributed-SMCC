nNodes=$2   
nThreads=$3
pyfile=$1
totalNodes=$((nNodes*nThreads));

prefix="experiment_results_output_data/ExpRG_8v32"
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
		cmd3="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGGB 1 ${seed}"
		cmd5="timeout 21600 mpirun -np ${totalNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGGB 1 ${seed}"
		
		if [ "$seed" == "42" ]; then
			echo $cmd1 # R-DASH
			# $cmd1
			
			echo $cmd3 # RandGreedI on $nNodes independent processing units with no parallelism
			# $cmd3

			echo $cmd5 # RandGreedI on $(nNodes * nThreads) independent processing units with no parallelism
			# $cmd5

		fi

    done
    
    echo "Data generated for seed = '${seed}'"
done

# Plot Fig 1,4 plots - Stored in plots/Fig1
cmd="bash plot_scripts/plot_FigApp_RG_8v32.sh"
echo $cmd
$cmd