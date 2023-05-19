//
//  main.cpp
//  PCA
//
//  Created by Salah on 06/05/2023.
//

#include <iostream>
#include <fstream>
#include <vector>
#include "parallelPCA.hpp"
#include "naivePCA.hpp"
#include "simulateData.hpp"
#include <chrono>
#include <iomanip>
#include <omp.h>


int main() {
    // Define the parameters for the simulation
    int cols = 10;
    double mean = 0.0;
    double std = 100.0;
    double scale = 100.0;

    // Define the number of rows and the step size
    int minRows = 1000;
    int stepSize = 10000;
    int iterations = 100;
    int maxRows = minRows + stepSize * iterations;

    // Define the row numbers as a grid from minRows to maxRows in steps of stepSize
    std::vector<int> rowNumbers;
    std::ofstream output;
    
    for (int rows = minRows; rows <= maxRows; rows += stepSize) {
        rowNumbers.push_back(rows);
    }
    output.open("execution_times.dat");

    //print the number of threads
    std::cout << "Number of threads: " << omp_get_max_threads() << std::endl;
    
    //print the table header
    std::cout << std::setw(10) << "Num Rows" << std::setw(15) << "Normal PCA" << std::setw(20) << "Parallel PCA" << std::endl;
    std::cout << "----------------------------------------------------" << std::endl;
    
    //run PCA for each rowNumber then print execution times
    for (int rows : rowNumbers) {
        std::vector<std::vector<double>> data = simulateData(rows, cols, mean, std, scale);

        std::vector<double> eigenvalues;
        std::vector<std::vector<double>> eigenvectors;
        
        double normalTime;
        double parallelTime;
        
        // run  benchmark PCA
        normalTime = naivePCA(data, eigenvalues, eigenvectors);

        // run parallel PCA
        parallelTime = parallelPCA(data, eigenvalues, eigenvectors);


        // print the execution times in the table
        std::cout << std::setw(10) << rows << std::setw(15) << normalTime << std::setw(20) << parallelTime << std::endl;
        
        //save data to file
        output << std::setw(10) << rows << std::setw(15) << normalTime << std::setw(20) << parallelTime << std::endl;
    }
    
    output.close();
    
    return 0;
}
