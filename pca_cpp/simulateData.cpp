//
//  simulateData.cpp
//  PCA
//
//  Created by Salah on 11/05/2023.
//

#include "simulateData.hpp"
#include <random>

std::vector<std::vector<double>> simulateData(int nRows, int nCols, double mean, double std, double scale) {
    std::vector<std::vector<double>> data(nRows, std::vector<double>(nCols));

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<double> distribution(mean, std);

    for (int i = 0; i < nRows; ++i) {
        for (int j = 0; j < nCols; ++j) {
            data[i][j] = distribution(gen) * scale;
        }
    }

    for (int j = 0; j < nCols; ++j) {
        double columnMean = 0.0;
        for (int i = 0; i < nRows; ++i) {
            columnMean += data[i][j];
        }
        columnMean /= nRows;

        for (int i = 0; i < nRows; ++i) {
            data[i][j] -= columnMean;
        }
    }

    return data;
}
