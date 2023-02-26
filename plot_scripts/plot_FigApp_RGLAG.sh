bash bash_scripts/mergeAllReps.sh

#IMAGESUMM
data='IMAGESUMM'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv

#IFM
data='INFLUENCEEPINIONS'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv

#YOUTUBE2000
data='YOUTUBE2000'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv

#BA
data='BA_100k'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp1/${data}_exp1_RandGreedI-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_RandGreedILAG-8-4.csv experiment_results_output_data/Exp1/${data}_exp1_DASH-8-4.csv



#BA
data='ba1m_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 

# #IFM
data='Youtube_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 

#REVMAX
data='Orkut_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 


# #IMAGESUMM
data='images_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/Exp2/${data}_exp2_RandGreedI-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_RandGreedILAG-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv 




# mogrify -format pdf -- plots/*.png
# rm plots/*.png

mkdir -p plots/FigApp/RGLAG
mkdir -p plots/FigApp/RGLAG/Exp1
mkdir -p plots/FigApp/RGLAG/Exp2

mv plots/*_exp1.pdf plots/FigApp/RGLAG/Exp1/
mv plots/*_exp2.pdf plots/FigApp/RGLAG/Exp2/
# mogrify -format pdf -- plots/*exp1.png
# rm plots/*exp1.png