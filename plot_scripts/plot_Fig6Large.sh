#BA
data='ba1m_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_PGBHeuristic-8-4.csv

# #IFM
data='Youtube_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_PGBHeuristic-8-4.csv

#REVMAX
data='Orkut_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_PGBHeuristic-8-4.csv


# #IMAGESUMM
data='images_sparse'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/Exp2/${data}_exp2_DASH-8-4.csv experiment_results_output_data/Exp2/${data}_exp2_PGBHeuristic-8-4.csv

# mogrify -format pdf -- plots/*.png
# rm plots/*.png

mkdir -p plots/Fig6Large/
mv plots/*.pdf plots/Fig6Large/
# mogrify -format pdf -- plots/*exp1.png
# rm plots/*exp1.png