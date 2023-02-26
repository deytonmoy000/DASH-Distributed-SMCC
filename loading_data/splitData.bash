#!/bin/bash

nNodesMin=$1
nNodesMax=$2
((nNodesMin *= 2))
declare -a objs=("youtube1m_split" "orkut3m_split" "images_sparse" "ba1m_sparse")

for i in ${!objs[@]};
do
	obj=${objs[$i]}
	nNodesMintmp=${nNodesMin}
	if [ "$obj" = "images_sparse" ] || [ "$obj" = "ba1m_sparse" ]; then
    	cmd="python3 split_npz.py data_exp2_split/${obj}_1_1.npz ${obj} 8"
		echo $cmd
		$cmd
	else
		while [[ ${nNodesMintmp} -le ${nNodesMax} ]] ; do
		    data=${nNodesMintmp}
			
			cmd="python3 split_npz.py data_exp2_split/${obj}_1_1.npz ${obj} ${data}"
			echo $cmd
			$cmd

			if [ "$data" -eq ${nNodesMax} ]; then
			    cmd="cp data_exp2_split/${obj}_${data}_*.npz data_exp3_split/"
			else
			    cmd="mv data_exp2_split/${obj}_${data}_*.npz data_exp3_split/"
			
			fi
			echo $cmd
			$cmd
			((nNodesMintmp *= 2))
		done
	fi
done

cmd="mv data_exp2_split/*1_1.npz data_exp3_split/"
echo $cmd
$cmd

echo "Data Assignment Complete"
echo "Data copying ..."


for k in `seq ${nNodesMax}`;
do
	# obj=${objsNew[$i]}
	rm -f machine${k}Data/data_exp2_split/*.npz
	rm -f machine${k}Data/data_exp3_split/*.npz
	cmd="mv data_exp2_split/*_${k}.npz machine${k}Data/data_exp2_split/"
	echo $cmd
	$cmd
	cmd2="mv data_exp3_split/*_split*_${k}.npz machine${k}Data/data_exp3_split/"
	echo $cmd2
	$cmd2	
done

echo "Renaming Files"
for k in `seq ${nNodesMax}`;
do
	for f in machine${k}Data/data_exp2_split/*.npz; do mv "$f" "$(echo "$f" | sed s/_${k}.npz/.npz/)"; done
	
	for f in machine${k}Data/data_exp3_split/*.npz; do mv "$f" "$(echo "$f" | sed s/_${k}.npz/.npz/)"; done
	
done
