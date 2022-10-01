//
// Created by kazem on 2022-08-26.
//


#ifndef LOOP_DEPENDENCE_UTILS_H
#define LOOP_DEPENDENCE_UTILS_H

#include <random>


void rand_test_matrix(int m, double *mat){
 std::random_device rd;
 std::mt19937 mt(rd());
 std::uniform_real_distribution<double> dist(.0, 1.0);
 for (int i = 0; i < m; ++i) {
  for (int j = 0; j <= i; ++j) {
   mat[i*m+j] = dist(mt);
   if(i == j)
    mat[i*m+j] += 10;//for numerical stability
  }
 }
}



#endif //LOOP_DEPENDENCE_UTILS_H
