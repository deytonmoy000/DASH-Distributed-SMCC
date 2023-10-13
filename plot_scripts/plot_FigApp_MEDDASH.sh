
#IMAGESUMM
data='IMAGESUMM'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv

#IFM
data='INFLUENCEEPINIONS'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv

#YOUTUBE2000
data='YOUTUBE2000'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv

#BA
data='BA_100k'
python3 plot_scripts/plot_util.py v 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv
python3 plot_scripts/plot_util.py t 1 experiment_results_output_data/ExpApp/${data}_exp1_MEDDASH-8-4.csv experiment_results_output_data/ExpApp/${data}_exp1_DASH-8-4.csv

# mogrify -format pdf -- plots/*.png
# rm plots/*.png

mkdir -p plots/FigApp/MEDDASH
mv plots/*.pdf plots/FigApp/MEDDASH/