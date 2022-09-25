//
// Created by kazem on 2022-08-26.
//

#ifndef LOOP_DEPENDENCE_UTILS_H
#define LOOP_DEPENDENCE_UTILS_H

void rand_test_matrix(int m, double *mat){
 for (int i = 0; i < m; ++i) {
  for (int j = 0; j <= i; ++j) {
   mat[i*m+j] = 1.0;
  }
 }
}


#endif //LOOP_DEPENDENCE_UTILS_H
