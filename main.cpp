//
// Created by kazem on 2022-08-26.
//


#include "aggregation/def.h"
#include "sympiler/sparse_blas_lib.h"
#include "aggregation/test_utils.h"
#include "aggregation/utils.h"
#include "aggregation/sparse_utilities.h"
#include "aggregation/sparse_io.h"
#include "aggregation/lbc.h"
#include "aggregation/hdagg.h"

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
int test_sparse(int argc, char *argv[]);
int test_sparse_mkl(int argc, char *argv[]);


int main(int argc, char *argv[]){
 int dense_sw = 0;
 if(argc > 1){
  dense_sw = atoi(argv[1]);
 }
 if(dense_sw == 0){
  std::cout<<"Dense,";
  test_dense(argc, argv);
 }
 else if(dense_sw == 1){
  std::cout<<"Sparse,";
  test_sparse(argc-1, argv+1);
 } else{
  std::cout<<"Sparse MKL,";
  test_sparse_mkl(argc-1, argv+1);
 }
 std::cout<<"\n";
 return 1;
}


int test_dense(int argc, char *argv[]){
 assert(argc>2);
 int m = atoi(argv[2]);
 int num_threads = atoi(argv[3]);
// omp_set_num_threads(num_threads);
//#ifdef MKL
// MKL_Set_Num_Threads(num_threads);
// MKL_Domain_Set_Num_Threads(num_threads, MKL_DOMAIN_BLAS);
//#endif
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


int test_sparse(int argc, char *argv[]){
 CSC *L1_csc, *A = NULLPNTR;
 CSR *L1_csr;
 size_t n;
 int num_threads = 6;
 int *perm;
 std::string matrix_name;
 std::vector<timing_measurement> time_array;
 if (argc < 2) {
  PRINT_LOG("Not enough input args, switching to random mode.\n");
  n = 16;
  double density = 0.2;
  matrix_name = "Random_" + std::to_string(n);
  A = random_square_sparse(n, density);
  if (A == NULLPNTR)
   return -1;
  L1_csc = make_half(A->n, A->p, A->i, A->x);
 } else {
  std::string f1 = argv[1];
  matrix_name = f1;
  L1_csc = read_mtx(f1);
  if (L1_csc == NULLPNTR)
   return -1;
  n = L1_csc->n;
 }
 if(argc >= 3)
  num_threads = atoi(argv[2]);
 //omp_set_num_threads(num_threads);
 int use_HDAGG = 1;
 if(argc >= 4)
  use_HDAGG = atoi(argv[3]);
 /// Re-ordering L matrix
#ifdef METIS
 //We only reorder L since dependency matters more in l-solve.
 //perm = new int[n]();
 CSC *L1_csc_full = make_full(L1_csc);
 delete L1_csc;
 metis_perm_general(L1_csc_full, perm);
 L1_csc = make_half(L1_csc_full->n, L1_csc_full->p, L1_csc_full->i,
                    L1_csc_full->x);
 CSC *Lt = transpose_symmetric(L1_csc, perm);
 CSC *L1_ord = transpose_symmetric(Lt, NULLPNTR);
 delete L1_csc;
 L1_csc = L1_ord;
 delete Lt;
 delete L1_csc_full;
 delete[]perm;
#endif
 L1_csr = csc_to_csr(L1_csc);
 auto *y = new double[n](), *x = new double[n]();
 auto *y_cpy = new double[n]();
 //////// SpMV
 std::vector<timing_measurement> t_spmv_array;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(x, n, RVAL);
  timing_measurement t_spmv; t_spmv.start_timer();
  sym_lib::spmv_csr_parallel(n, L1_csr->p, L1_csr->i, L1_csr->x, x, y);
  t_spmv.measure_elapsed_time();
  t_spmv_array.push_back(t_spmv);
 }
 auto t_spmv_sec = time_median(t_spmv_array).elapsed_time;
 std::copy(y, y+n, y_cpy);
 //////// SpTRSV
 std::vector<timing_measurement> t_sptrsv_array, t_sptrsv_seq_array;
 int final_level_no, *fina_level_ptr=NULLPNTR, *final_part_ptr=NULLPNTR, *final_node_ptr=NULLPNTR;
 int coarse_level_no;
 std::vector<int> coarse_level_ptr, coarse_part_ptr, coarse_node_ptr;

 for (int i = 0; i < NTIMES; ++i) {
  std::copy(y_cpy, y_cpy+n, y);
  timing_measurement t_sptrsv; t_sptrsv.start_timer();
   sptrsv_csr(n, L1_csr->p, L1_csr->i, L1_csr->x, y);
  t_sptrsv.measure_elapsed_time();
  t_sptrsv_seq_array.push_back(t_sptrsv);
 }
 auto t_sptrsv_seq_sec = time_median(t_sptrsv_seq_array).elapsed_time;
 for (int i = 0; i < n; ++i) {
  if(std::abs(y[i] - RVAL) > 1e-5){
   std::cout<<"SEQ ERROR in "<<i<< " diff "<< std::abs(y[i] - RVAL)<<",";
   return 0;
  }
 }


 int part_no;
 int lp_=6, cp_=10, ic_=4;
 auto *cost = new double[n]();
 for (int i = 0; i < n; ++i) {
  cost[i] = L1_csr->p[i+1] - L1_csr->p[i];
 }
 /// SpTRSV Inspector
 timing_measurement inspect_time; inspect_time.start_timer();
 if(!use_HDAGG){
  get_coarse_levelSet_DAG_CSC_tree(n, L1_csr->p, L1_csr->i,
                                   L1_csr->stype,
                                   final_level_no,
                                   fina_level_ptr,part_no,
                                   final_part_ptr,final_node_ptr,
                                   lp_,cp_, ic_, cost);
 } else{
  HDAGG::build_coarsened_level_parallel(n, L1_csc->p, L1_csc->i, lp_,
                                 coarse_level_no, coarse_level_ptr, coarse_part_ptr, coarse_node_ptr);
 }
 inspect_time.measure_elapsed_time();
 /// SpTRSV Executor
 for (int i = 0; i < NTIMES; ++i) {
  std::copy(y_cpy, y_cpy+n, y);
  timing_measurement t_sptrsv; t_sptrsv.start_timer();
  if(!use_HDAGG) {
   sptrsv_csr_lbc(n, L1_csr->p, L1_csr->i, L1_csr->x, y,
                  final_level_no, fina_level_ptr,
                  final_part_ptr, final_node_ptr);
  } else {
   sptrsv_csr_lbc(n, L1_csr->p, L1_csr->i, L1_csr->x, y,
                  coarse_level_no, coarse_level_ptr.data(),
                  coarse_part_ptr.data(), coarse_node_ptr.data());
  }
  t_sptrsv.measure_elapsed_time();
  t_sptrsv_array.push_back(t_sptrsv);
 }
 auto t_sptrsv_sec = time_median(t_sptrsv_array).elapsed_time;

 //sptrsv_csr(n, L1_csr->p, L1_csr->i, L1_csr->x, y);
 /// Logging
 std::cout<<n<<","<<n<<","<<L1_csr->nnz<<","<<t_spmv_sec<<","<<t_sptrsv_sec<<","<<inspect_time.elapsed_time<<",";
 std::cout<<matrix_name<<","<< ( use_HDAGG ? "HDAGG" : "LBC") <<","<<t_sptrsv_seq_sec<<",";
 std::cout<<num_threads<<",";
 /// Testing
 for (int i = 0; i < n; ++i) {
  if(std::abs(y[i] - RVAL) > 1e-5){
   std::cout<<"ERROR in "<<i<< " diff "<< std::abs(y[i] - RVAL)<<",";
   return 0;
  }
 }
 std::cout<<"PASSED,";

 delete A;
 delete L1_csc;
 delete L1_csr;
 delete []x;
 delete []y;
 delete []cost;
 delete []fina_level_ptr;
 delete []final_part_ptr;
 delete []final_node_ptr;
 return 1;
}


int test_sparse_mkl(int argc, char *argv[]){
 CSC *L1_csc, *A = NULLPNTR;
 CSR *L1_csr;
 size_t n;
 int num_threads = 6;
 int *perm;
 std::string matrix_name;
 std::vector<timing_measurement> time_array;
 if (argc < 2) {
  PRINT_LOG("Not enough input args, switching to random mode.\n");
  n = 16;
  double density = 0.2;
  matrix_name = "Random_" + std::to_string(n);
  A = random_square_sparse(n, density);
  if (A == NULLPNTR)
   return -1;
  L1_csc = make_half(A->n, A->p, A->i, A->x);
 } else {
  std::string f1 = argv[1];
  matrix_name = f1;
  L1_csc = read_mtx(f1);
  if (L1_csc == NULLPNTR)
   return -1;
  n = L1_csc->n;
 }
 if(argc >= 3)
  num_threads = atoi(argv[2]);
 //omp_set_num_threads(num_threads);

 /// Re-ordering L matrix
#ifdef METIS
 //We only reorder L since dependency matters more in l-solve.
 //perm = new int[n]();
 CSC *L1_csc_full = make_full(L1_csc);
 delete L1_csc;
 metis_perm_general(L1_csc_full, perm);
 L1_csc = make_half(L1_csc_full->n, L1_csc_full->p, L1_csc_full->i,
                    L1_csc_full->x);
 CSC *Lt = transpose_symmetric(L1_csc, perm);
 CSC *L1_ord = transpose_symmetric(Lt, NULLPNTR);
 delete L1_csc;
 L1_csc = L1_ord;
 delete Lt;
 delete L1_csc_full;
 delete[]perm;
#endif
 L1_csr = csc_to_csr(L1_csc);
 auto *y = new double[n](), *x = new double[n]();
 auto *y_cpy = new double[n]();
 //////// SpMV
 std::vector<timing_measurement> t_spmv_array;
 int num_calls = 1000;
 sparse_matrix_t csrL, csrA;
 struct matrix_descr descrL;
 descrL.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
 descrL.mode = SPARSE_FILL_MODE_LOWER;
 descrL.diag = SPARSE_DIAG_NON_UNIT;
 auto statA = mkl_sparse_d_create_csr(&csrA, SPARSE_INDEX_BASE_ZERO,
                                      n,  // number of rows
                                      n,  // number of cols
                                      L1_csr->p,
                                      L1_csr->p + 1,
                                      L1_csr->i,
                                      L1_csr->x);
 /// SpMV Inspector
 auto stat = mkl_sparse_set_mv_hint(csrA, SPARSE_OPERATION_NON_TRANSPOSE,
                               descrL, num_calls);
 if (stat != SPARSE_STATUS_SUCCESS) {
  printf("analysis failed with %d\n",stat);
  return -1;
 }
 if (mkl_sparse_optimize(csrA) != SPARSE_STATUS_SUCCESS) {
  printf("optimization failed with ");
 }
 double alpha =1.0, beta=0.0;
 for (int i = 0; i < NTIMES; ++i) {
  std::fill_n(x, n, RVAL);
  timing_measurement t_spmv; t_spmv.start_timer();
  auto stats = mkl_sparse_d_mv(SPARSE_OPERATION_NON_TRANSPOSE,
                               alpha,
                               csrA,
                               descrL,
                               x,
                               beta,
                               y);
  t_spmv.measure_elapsed_time();
  if(stats != SPARSE_STATUS_SUCCESS){
   std::cout<<"mv failed";
   return -1;
  }
  t_spmv_array.push_back(t_spmv);
 }
 auto t_spmv_sec = time_median(t_spmv_array).elapsed_time;
 std::copy(y, y+n, y_cpy);
 //////// SpTRSV
 auto statl = mkl_sparse_d_create_csr(&csrL, SPARSE_INDEX_BASE_ZERO,
                                      n,  // number of rows
                                      n,  // number of cols
                                      L1_csr->p,
                                      L1_csr->p + 1,
                                      L1_csr->i,
                                      L1_csr->x);
 std::vector<timing_measurement> t_sptrsv_array, t_sptrsv_seq_array;
 timing_measurement inspect_time; inspect_time.start_timer();
 /// SpTRSV Inspector
 stat = mkl_sparse_set_sv_hint(csrL, SPARSE_OPERATION_NON_TRANSPOSE,
                               descrL, num_calls);
 if (stat != SPARSE_STATUS_SUCCESS) {
  printf("analysis failed with %d\n", stat);
  //return -1;
 }
 if (mkl_sparse_optimize(csrL) != SPARSE_STATUS_SUCCESS) {
  printf("optimization failed with %d;");
 }
 inspect_time.measure_elapsed_time();
 /// SpTRSV Executor
 for (int i = 0; i < NTIMES; ++i) {
  //std::copy(y_cpy, y_cpy+n, y);
  std::fill_n(y, n, 0.0);
  timing_measurement t_sptrsv; t_sptrsv.start_timer();
  auto stat = mkl_sparse_d_trsv(SPARSE_OPERATION_NON_TRANSPOSE,
                                1,
                                csrL,
                                descrL,
                                y_cpy,
                                y);
  t_sptrsv.measure_elapsed_time();
  if(stat != SPARSE_STATUS_SUCCESS){
   std::cout<<"sv failed";
   return -1;
  }
  t_sptrsv_array.push_back(t_sptrsv);
  std::cout<<stat;
 }
 auto t_sptrsv_sec = time_median(t_sptrsv_array).elapsed_time;

 //sptrsv_csr(n, L1_csr->p, L1_csr->i, L1_csr->x, y);
 /// Logging
 std::cout<<n<<","<<n<<","<<L1_csr->nnz<<","<<t_spmv_sec<<","<<t_sptrsv_sec<<","<<inspect_time.elapsed_time<<",";
 std::cout<<matrix_name<<","<< "MKL"<<",";
 std::cout<<num_threads<<",";
 /// Testing
 for (int i = 0; i < n; ++i) {
  if(std::abs(y[i] - RVAL) > 1e-5){
   std::cout<<"ERROR in "<<i<< " diff "<< std::abs(y[i] - RVAL)<<",";
   return 0;
  }
 }
 std::cout<<"PASSED,";

 delete A;
 delete L1_csc;
 delete L1_csr;
 delete []x;
 delete []y;
 return 1;
}