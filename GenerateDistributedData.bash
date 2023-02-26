#!/bin/bash
nThreads=$1
nNodes=$2

cd  loading_data/ 

# Deleting all previous data
rm -rf machine*
rm -rf downloaded_data
rm -rf imageData
rm -rf cifar-10-batches-bin
rm -rf data_exp2_split
rm -rf data_exp3_split
rm -f preproc_images

echo "Deleted Existing Data"

# Creating the clean repositories
mkdir downloaded_data
mkdir imageData
mkdir data_exp2_split
mkdir data_exp3_split

for k in `seq ${nNodes}`; do
	mkdir machine${k}Data
	cp -r *_split machine${k}Data/
done
echo "Created New Directories"

echo "cd downloaded_data"
cd downloaded_data

echo "rm *.tar.*"
rm *.tar.*

echo "wget http://konect.cc/files/download.tsv.orkut-links.tar.bz2"
wget http://konect.cc/files/download.tsv.orkut-links.tar.bz2

echo "wget http://konect.cc/files/download.tsv.com-youtube.tar.bz2"
wget http://konect.cc/files/download.tsv.com-youtube.tar.bz2

echo "wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

echo "tar -xvf download.tsv.orkut-links.tar.bz2"
tar -xvf download.tsv.orkut-links.tar.bz2

echo "tar -xvf download.tsv.com-youtube.tar.bz2"
tar -xvf download.tsv.com-youtube.tar.bz2

echo "tar -xvf cifar-10-binary.tar.gz"
tar -xvf cifar-10-binary.tar.gz


echo "cd .."
cd ..

#  Generate Image Distributed Data
echo "mv downloaded_data/cifar-10-batches-bin ."
mv downloaded_data/cifar-10-batches-bin .

echo "make preproc_images"
make preproc_images

cmd="./preproc_images ${nThreads}"
echo $cmd
$cmd

echo "cat imageData/*.csv > imageData/images_50K_mat.csv"
cat imageData/*.csv > imageData/images_50K_mat.csv

rm -rf imageData/images_t*

echo "python3 dense_to_sparse_npz.py imageData/images_50K_mat.csv data_exp2_split/images_sparse_1_1.npz"
python3 dense_to_sparse_npz.py imageData/images_50K_mat.csv data_exp2_split/images_sparse_1_1.npz


#  Generate InfluenceMax Distributed Data
echo "sed -e 's/\s/,/g' downloaded_data/com-youtube/out.com-youtube > downloaded_data/youtube1M_edgelist.csv"
sed -e 's/\s/,/g' downloaded_data/com-youtube/out.com-youtube > downloaded_data/youtube1M_edgelist.csv

echo "sed -i '1,2d' downloaded_data/youtube1M_edgelist.csv"
sed -i '1,2d' downloaded_data/youtube1M_edgelist.csv

echo "sed -i '1s/^/FromNodeId,ToNodeId\n/' downloaded_data/youtube1M_edgelist.csv"
sed -i '1s/^/FromNodeId,ToNodeId\n/' downloaded_data/youtube1M_edgelist.csv

echo "python3 edgelist_to_sparse_npz.py downloaded_data/youtube1M_edgelist.csv data_exp2_split/youtube1m_split_1_1.npz"
python3 edgelist_to_sparse_npz.py downloaded_data/youtube1M_edgelist.csv data_exp2_split/youtube1m_split_1_1.npz


#  Generate RevenueMax Distributed Data
echo "sed -e 's/\s/,/g' downloaded_data/orkut-links/out.orkut-links > downloaded_data/orkut3M_edgelist.csv"
sed -e 's/\s/,/g' downloaded_data/orkut-links/out.orkut-links > downloaded_data/orkut3M_edgelist.csv

echo "sed -i '1,2d' downloaded_data/orkut3M_edgelist.csv"
sed -i '1,2d' downloaded_data/orkut3M_edgelist.csv

echo "sed -i '1s/^/FromNodeId,ToNodeId\n/' downloaded_data/orkut3M_edgelist.csv"
sed -i '1s/^/FromNodeId,ToNodeId\n/' downloaded_data/orkut3M_edgelist.csv

echo "python3 edgelist_to_sparse_npz.py downloaded_data/orkut3M_edgelist.csv data_exp2_split/orkut3m_split_1_1.npz"
python3 edgelist_to_sparse_npz.py downloaded_data/orkut3M_edgelist.csv data_exp2_split/orkut3m_split_1_1.npz


#  Generate MaxCover (BA) Distributed Data
echo "python3 MatrixGenSparse.py"
python3 MatrixGenSparse.py


# Create a Backup
mkdir -p data_exp2_split/BKP
echo "mv data_exp2_split/*.npz data_exp2_split/BKP/"
cp data_exp2_split/*.npz data_exp2_split/BKP/


# Split the data and place them in corresponding machines folder
cmd2="bash splitData.bash 1 ${nNodes}"
echo $cmd2
$cmd2

rm -rf downloaded_data
rm -rf imageData
rm -rf cifar-10-batches-bin