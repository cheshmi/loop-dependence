//
// Created by kazem on 2022-08-26.
//

#include "utils.h"



#include "def.h"
#include "sparse_blas_lib.h"
#include "test_utils.h"
#include "sparse_utilities.h"
#include "sparse_io.h"
#include "lbc.h"

#include <iostream>

#ifdef METIS
#include "metis_interface.h"
#endif

#include "cblas.h"

#include <omp.h>

void test_dense();
int test_sparse(int argc, char *argv[]);


int main(int argc, char *argv[]){

 test_dense();
 test_sparse(argc, argv);
 return 1;
}


void test_dense(){
 int m = 50;
 auto matrix = new double [m*m]();
 rand_test_matrix(m, matrix);
 auto vec = new double[m]();
 auto y = new double[m]();
 std::fill_n(vec, m, 0.5);
 cblas_dgemv(CblasRowMajor, CblasNoTrans,
             m,  m,
             1.,              // alpha
             matrix, m,
             vec, 1,
             0.,              // beta
             y, 1);
 for (int i = 0; i < m; ++i) {
  std::cout<<y[i]<<", ";
 }
 std::cout<<"\n";

 cblas_dtrsm(CblasRowMajor, CblasRight, CblasLower, CblasConjTrans,
             CblasNonUnit,
             m, m, 1,
             matrix, m, y, m);

 for (int i = 0; i < m; ++i) {
  std::cout<<y[i]<<", ";
 }
 std::cout<<"\n";
}

using namespace sym_lib;

int test_sparse(int argc, char *argv[]){
 CSC *L1_csc, *A = NULLPNTR;
 CSR *L1_csr;
 size_t n;
 int num_threads = 6;
 int p2 = -1, p3 = 4000; // LBC params
 int header = 0;
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
  p2 = atoi(argv[2]);
 omp_set_num_threads(num_threads);
 if(argc >= 4)
  p3 = atoi(argv[3]);
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
 double *y = new double[n](), *x = new double[n]();

 //////// SpMV
 std::fill_n(x, n, .1);
 sym_lib::spmv_csr_parallel(n, L1_csr->p, L1_csr->i, L1_csr->x, x, y);

 //////// SpTRSV
 int final_level_no, *fina_level_ptr, *final_part_ptr, *final_node_ptr;
 int part_no;
 int lp_, cp_, ic_;
 auto *cost = new double[n]();
 for (int i = 0; i < n; ++i) {
  cost[i] = L1_csr->p[i+1] - L1_csr->p[i];
 }
 /// SpTRSV Inspector
 get_coarse_levelSet_DAG_CSC_tree(n, L1_csr->p, L1_csr->i,
                                  L1_csr->stype,
                                  final_level_no,
                                  fina_level_ptr,part_no,
                                  final_part_ptr,final_node_ptr,
                                  lp_,cp_, ic_, cost);

 /// SpTRSV Executor
 sptrsv_csr_lbc(n, L1_csr->p, L1_csr->i, L1_csr->x, y,
                final_level_no, fina_level_ptr,
                final_part_ptr, final_node_ptr);

 print_vec("Y: ", 0, n, y);

 delete A;
 delete L1_csc;
 delete L1_csr;
 delete []x;
 delete []y;
 delete []cost;
}