#include <iostream>
#include <string>
#include "./include/cifar/cifar10_reader.hpp"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <cstdint>
#include <memory>
#include <thread>

#include <sstream>
#include <iomanip>
#include <cmath>

using namespace std;
using namespace cifar;

/*
* This preprocessing is used to tranform the CIFAR 10 data to
* an interpretable similarity adj Matrix 
*/

auto dataset = cifar::read_dataset<std::vector, std::vector, uint8_t, uint8_t>();

double cosine_similarity(vector<uint8_t> A, vector<uint8_t> B)
{
    double dot = 0.0, denom_a = 0.0, denom_b = 0.0 ;
    for(unsigned int i = 0; i < A.size(); ++i) {
        size_t A_i = static_cast<size_t>(A[i]);
        size_t B_i = static_cast<size_t>(B[i]);
        dot += A_i * B_i ;
        denom_a += A_i * A_i ;
        denom_b += B_i * B_i ;
    }
    return (dot / (sqrt(denom_a) * sqrt(denom_b)));
}

void parallel_preproc_img(string fname, size_t StartElement, size_t EndElement, size_t nThreads) {
  if (nThreads == 1) {
    string newFname = fname + ".csv";
    ofstream ifile ( newFname.c_str(), ios::out );
    fstream myfile;
    myfile.open(newFname.c_str(),fstream::out);
    vector<double> simI(dataset.training_images.size(), 0.000000);
    vector<vector<double>> simMat(dataset.training_images.size(), simI);
    for(unsigned int i = StartElement; i < EndElement; ++i) {
        vector<uint8_t> A = dataset.training_images[i];
        for(unsigned int j = 0; j < dataset.training_images.size()-1; ++j)
        {
            vector<uint8_t> B = dataset.training_images[j];
            long double wht = cosine_similarity(A, B);
            myfile <<std::fixed << setprecision(6) << wht<<",";
        }
        vector<uint8_t> B = dataset.training_images[dataset.training_images.size()-1];
        long double wht = cosine_similarity(A, B);
        myfile << std::fixed << setprecision(6) << wht<< std::endl;
        if(i%100==0 && EndElement==50000)
            cout << "Adjacent matrix row printed for Node-"<<i<< endl;
    }
    myfile.close();
  }
  else {
    thread* wThreads = new thread[ nThreads ];
    size_t StartElement = 0;
    size_t EndElement;
    string tmpFname = fname;
    for (size_t i =0; i < nThreads; ++i) {
      if (i < nThreads - 1)
        EndElement = StartElement + dataset.training_images.size() / nThreads;
      else
        EndElement = dataset.training_images.size();
      string fname = "";
      if(i < 9)
          fname = tmpFname + "_0"+std::to_string(i+1);
      else
          fname = tmpFname + "_"+std::to_string(i+1);
      wThreads[i] = thread( &parallel_preproc_img,
                            fname,
                            StartElement,
                            EndElement,
                            1 
                          );
                
                
      StartElement += dataset.training_images.size() / nThreads;

    }
    for (size_t i = 0; i < nThreads; ++i) {
      wThreads[i].join();
    }
    delete [] wThreads;
  }
}

int main( int argc, char** argv ) {
    if (argc < 2) {
        cout << "Usage: " << argv[0] << " <number of Threads>" << endl;
        cout << "CIFAR-10 Images are simplified, then similarity Adj Matrix written to output file" << endl;
        exit(1);
    }
    size_t nThreads = 1;
    nThreads = std::stoi(argv[1]);
    cout << "Writing temp text file..." << endl;
    // // string outputTxtFName( "./images_tmp_0.txt" );

    // ofstream ifile ( outputTxtFName.c_str(), ios::out );
    
    cout << "Data Size:" << dataset.training_images.size() << endl;
    parallel_preproc_img("./imageData/images_tmp", 0, dataset.training_images.size(), nThreads);
}