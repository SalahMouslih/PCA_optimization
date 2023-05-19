//
//  naivePCA.hpp
//  PCA
//
//  Created by Salah on 08/05/2023.
//

#ifndef naivePCA_hpp
#define naivePCA_hpp

#include <stdio.h>
#include <vector>

double naivePCA(const std::vector<std::vector<double>>& data, std::vector<double>& eigenvalues, std::vector<std::vector<double>>& eigenvectors);

#endif /* naivePCA_hpp */
