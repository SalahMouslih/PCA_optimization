//
//  parallelPCA.hpp
//  PCA
//
//  Created by Salah on 08/05/2023.
//

#ifndef parallelPCA_hpp
#define parallelPCA_hpp

#include <stdio.h>
#include <vector>

double parallelPCA(const std::vector<std::vector<double>>& data, std::vector<double>& eigenvalues, std::vector<std::vector<double>>& eigenvectors);

#endif /* parallelPCA_hpp */
