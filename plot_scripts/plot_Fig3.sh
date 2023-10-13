# #IFM
data='Youtube_sparse'
python3 plot_scripts/plot_util_exp3.py t 1 experiment_results_output_data/Exp3/${data}_exp3_DASH-1-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-2-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-4-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-8-4.csv 

#REVMAX
data='Orkut_sparse'
python3 plot_scripts/plot_util_exp3.py t 1 experiment_results_output_data/Exp3/${data}_exp3_DASH-1-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-2-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-4-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-8-4.csv 

#REVMAX
data='ba1m_sparse'
python3 plot_scripts/plot_util_exp3.py t 1 experiment_results_output_data/Exp3/${data}_exp3_DASH-2-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-4-4.csv experiment_results_output_data/Exp3/${data}_exp3_DASH-8-4.csv 


# mogrify -format pdf -- plots/*.png
# rm plots/*.png

mkdir -p plots/Fig3/
mv plots/*.pdf plots/Fig3/
# mogrify -format pdf -- plots/*exp1.png
# rm plots/*exp1.png