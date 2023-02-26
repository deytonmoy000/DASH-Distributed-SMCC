CPPFLAGS=-std=c++11 -Wall -O2 -l pthread


all: preproc_images #revmax tweetSumm imgSumm caliTrf infMax movRec preproc preproc_tweet preproc_images preproc_CaliTraf preproc_rating

preproc_images: preprocess_images.cpp include/cifar/cifar10_reader.hpp 
	g++ preprocess_images.cpp -o preproc_images  ${CPPFLAGS}
clean:
	rm preproc_images
	
