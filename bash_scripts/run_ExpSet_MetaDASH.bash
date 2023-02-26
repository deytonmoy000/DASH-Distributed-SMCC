nNodes=$2   
nThreads=$3
pyfile=$1

prefix="experiment_results_output_data/"
under="_"
suffix=".csv"

declare -a objs=("IS" "IFM3" "RVM3" "BA")

for i in ${!objs[@]};
do
    obj=${objs[$i]}
    cmd1="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} DASH ${nThreads}"
    cmd2="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} RGGB ${nThreads}"
    cmd3="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} BCG ${nThreads}"    
    cmd4="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} MEDRG ${nThreads}"
    cmd5="timeout 21600 mpirun -np ${nNodes} --hostfile ../nodesFileIPnew --bind-to none python3 -W ignore ${pyfile} ${obj} MEDDASH ${nThreads}"
    
    echo $cmd1 # R-DASH
    $cmd1
    
    echo $cmd2 # RandGreedI
    $cmd2
    
    # echo $cmd3 # BicCriteriaGreedy
    # $cmd3

    # echo $cmd4 # MED+RG
    # $cmd4

    echo $cmd5 # MED+DASH
    $cmd5

    echo "Data generated for '${data}'"
done

# Plot Fig 7 (Appendix) - Stored in plots/FigApp/MEDDASH
cmd="bash plot_scripts/plot_FigApp_MEDDASH.sh "
echo $cmd
$cmd