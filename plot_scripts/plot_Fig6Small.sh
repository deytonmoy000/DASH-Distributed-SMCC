bash bash_scripts/mergeAllReps.sh

#BA
data='BA_100k'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_PGB-8-4.csv

#IFM
data='INFLUENCEEPINIONS'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_PGB-8-4.csv

#YOUTUBE2000
data='YOUTUBE2000'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_PGB-8-4.csv


#IMAGESUMM
data='IMAGESUMM'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_PGB-8-4.csv


# mogrify -format pdf -- plots/*.png
# rm plots/*.png

mkdir -p plots/Fig6Small
mv plots/*.pdf plots/Fig6Small/