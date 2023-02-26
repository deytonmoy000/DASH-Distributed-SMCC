cd experiment_results_output_data/Exp1/
rm *.csv  
nReps=5
for k in `seq ${nReps}`;
do
	for f in experiment_results_output_data/Exp1/Rep${k}_New/*.csv; do mv "$f" "$(echo "$f" | sed s/8-4.csv/8-4_Rep${k}.csv/)"; done
	cp Rep${k}_New/*Rep${k}.csv .
done

prefix="experiment_results_output_data"
under="_"
suffix=".csv"

declare -a objs=("DASH" "TDASH" "GDASH" "PGB" "RandGreedI" "BiCriteriaGreedy" "DD" "ParallelAlgoGreedy" "RandGreedILAG")

declare -a datas=("BA_100k" "INFLUENCEEPINIONS" "YOUTUBE2000" "IMAGESUMM")

rm *8-4.csv

for i in ${!objs[@]};
do
	for j in ${!datas[@]};
	do
		obj=${objs[$i]}
		data=${datas[$j]}
		filenm="${data}_exp1_${obj}-*"
		for file in ${data}_exp1_${obj}-*; do cat $file >> ${data}_exp1_${obj}-8-4.csv; done
		
		sed -i '/f_of_S/d' ${data}_exp1_${obj}-8-4.csv
		if [ "$obj" = "PGB" ]; then
		    sed -i '1s/^/f_of_S,Time,SolSize,k,n,nNodes,nThreads,trial\n/' ${data}_exp1_${obj}-8-4.csv
		else
		    sed -i '1s/^/f_of_S,Time,TimeDist,TimePost,SolSize,k,n,nNodes,nThreads,trial\n/' ${data}_exp1_${obj}-8-4.csv
		fi
		
		
	done
	
done

rm *Rep*.csv
find . -size 0 -delete



cd ../../