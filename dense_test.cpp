//
// Created by kazem on 2022-10-01.
//

#include "aggregation/def.h"
#include "aggregation/utils.h"
#include "utils_loop_cd.h"
#include <iostream>

#ifdef METIS
#include "aggregation/metis_interface.h"
#endif
#ifdef MKL
#include "mkl.h"
#endif


#include <omp.h>

#define NTIMES 5
#define RVAL 0.5
using namespace sym_lib;

int test_dense(int argc, char *argv[]);


int main(int argc, char *argv[]){
  std::cout<<"Dense,";
  test_dense(argc, argv);
 std::cout<<"\n";
 return 1;
}


int test_dense(int argc, char *argv[]){
 assert(argc>2);
 int m = atoi(argv[1]);
 int num_threads = atoi(argv[2]);
 omp_set_num_threads(num_threads);
 mkl_set_num_threads(num_threads);
 mkl_set_num_threads_local(num_threads);
 mkl_set_dynamic(0);
 auto matrix = new double [m*m]();
 rand_test_matrix(m, matrix);
 auto vec = new double[m]();
 auto y = new double[m]();
 auto y_cpy = new double[m]();
 /// MV -> y = matrix * vec
 std::vector<timing_measurement> t_mv_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(vec, m, RVAL);
  timing_measurement t_mv; t_mv.start_timer();
  cblas_dgemv(CblasRowMajor, CblasNoTrans,
              m,  m,
              1.,              // alpha
              matrix, m,
              vec, 1,
              0.,              // beta
              y, 1);
  t_mv.measure_elapsed_time();
  t_mv_array.push_back(t_mv);
 }
 auto t_mv_sec = time_median(t_mv_array).elapsed_time;
 std::copy(y, y+m, y_cpy);
 /// TRSV -> y = matrix \ y
 std::vector<timing_measurement> t_ts_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::copy(y_cpy, y_cpy+m, y);
  timing_measurement t_ts; t_ts.start_timer();
  cblas_dtrsm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans,
              CblasNonUnit,
              m, 1, 1,
              matrix, m, y, 1);
  t_ts.measure_elapsed_time();
  t_ts_array.push_back(t_ts);
 }
 double t_sv_sec = time_median(t_ts_array).elapsed_time;
 /// Logging
 std::cout<<m<<","<<m<<","<<m*m<<","<<t_mv_sec<<","<<t_sv_sec<<","<<num_threads<<",";
 /// Testing
 for (int i = 0; i < m; ++i) {
  if(std::abs(y[i] - RVAL) > 1e-5){
   std::cout<<"ERROR in "<<i<< " diff "<< std::abs(y[i] - RVAL)<<",";
   return 0;
  }
 }
 std::cout<<"PASSED,";
 delete []y;
 delete []y_cpy;
 delete []vec;
 delete []matrix;
 return 1;
}