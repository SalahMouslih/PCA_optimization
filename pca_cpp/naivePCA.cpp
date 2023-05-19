//
//  naivePCA.cpp
//  PCA
//
//  Created by Salah on 08/05/2023.
//

#include "naivePCA.hpp"
#include <Eigen/Dense>

double naivePCA(const std::vector<std::vector<double>>& data, std::vector<double>& eigenvalues, std::vector<std::vector<double>>& eigenvectors) {
    const std::size_t numSamples = data.size();
    const std::size_t numFeatures = data[0].size();

    Eigen::MatrixXd covMatrix(numFeatures, numFeatures);
    Eigen::MatrixXd dataMatrix(numSamples, numFeatures);

    // Prepare data matrix
    for (std::size_t i = 0; i < numSamples; ++i) {
        for (std::size_t j = 0; j < numFeatures; ++j) {
            dataMatrix(i, j) = data[i][j];
        }
    }

    // Calculate covariance matrix
    auto covStartTime = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < numFeatures; ++i) {
        for (std::size_t j = 0; j <= i; ++j) {
            covMatrix(i, j) = 0.0;
            for (std::size_t k = 0; k < numSamples; ++k) {
                covMatrix(i, j) += dataMatrix(k, i) * dataMatrix(k, j);
            }
            covMatrix(j, i) = covMatrix(i, j);
        }
    }
    auto covEndTime = std::chrono::steady_clock::now();

    // Compute eigenvectors and eigenvalues
    auto eigStartTime = std::chrono::steady_clock::now();
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covMatrix);
    Eigen::VectorXd eigenvaluesVec = eigensolver.eigenvalues();
    Eigen::MatrixXd eigenvectorsMat = eigensolver.eigenvectors();
    auto eigEndTime = std::chrono::steady_clock::now();

    // Convert eigenvalues and eigenvectors to std::vector
    eigenvalues.resize(numFeatures);
    eigenvectors.resize(numFeatures, std::vector<double>(numFeatures));
    for (std::size_t i = 0; i < numFeatures; ++i) {
        eigenvalues[i] = eigenvaluesVec(i);
        for (std::size_t j = 0; j < numFeatures; ++j) {
            eigenvectors[i][j] = eigenvectorsMat(j, i);
        }
    }

    auto covDuration = std::chrono::duration_cast<std::chrono::microseconds>(covEndTime - covStartTime).count();
    auto eigDuration = std::chrono::duration_cast<std::chrono::microseconds>(eigEndTime - eigStartTime).count();
    
    return static_cast<double>(covDuration + eigDuration)/ 1'000'000.0;
}
