//
// Created by kazem on 2022-10-01.
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

#include <omp.h>

#define NTIMES 5
#define RVAL 0.5
using namespace sym_lib;

struct schedule_param{
 int initial_cut, coarsen_factor, num_wparts, bin_pack= false, post_order= false;
};
int build_coarsened_level(const int n, const int* Lp, const int* Li,
                                   const int cores,
                                   schedule_param& param,
                                   int& coarse_level_no,
                                   std::vector<int>& coarse_level_ptr,
                                   std::vector<int>& coarse_part_ptr,
                                   std::vector<int>& coarse_node_ptr
){
 int coarsen_factor, num_wparts=cores, i_cut=0;
 bool bin_pack=false;
 bool postOrder=false;//set to false manually here
 if(param.coarsen_factor == -1){
  sym_lib::lbc_config(n, Lp[n], cores, num_wparts, coarsen_factor, i_cut, bin_pack);
 } else{
  coarsen_factor = param.coarsen_factor;
  num_wparts = param.num_wparts;
  i_cut = param.initial_cut;
  bin_pack = param.bin_pack;
  postOrder = param.post_order;
 }
 if(coarsen_factor > 1){
  std::vector<int> level_ptr;
  std::vector<int> level_set, node_to_level, WM;
  level_ptr.resize(n + 1);
  level_set.resize(n);
  node_to_level.resize(n);
  int n_levels = HDAGG::build_levelSet_CSC(n, Lp, Li, level_ptr.data(), level_set.data(),
                                           node_to_level.data() );
  for (int i = 0; i < n_levels;) {
   WM.push_back(i);
   i+=coarsen_factor;
   if(i >= n_levels){
    WM.push_back(n_levels);
    break;
   }
  }
  HDAGG::computeSchedule(cores, n, Lp, Li, level_ptr.data(), level_set.data(),
                         node_to_level.data(), WM, {{0}},
                         coarse_level_no, coarse_level_ptr, coarse_part_ptr, coarse_node_ptr,
                         bin_pack, postOrder);
 } else{ // fall back to basic wavefront for very sparse DAGs
  std::vector<int> node_to_level;
  coarse_part_ptr.resize(n + 1);
  coarse_node_ptr.resize(n);
  node_to_level.resize(n);
  coarse_level_no = HDAGG::build_levelSet_CSC(n, Lp, Li, coarse_part_ptr.data(),
                                              coarse_node_ptr.data(), node_to_level.data());
  coarse_level_ptr.resize( coarse_level_no + 1);
  for (int i = 0; i < coarse_level_no+1; ++i) {
   coarse_level_ptr[i] = i;
  }
 }
 return 1;
}


int test_sparse(int argc, char *argv[]);


int main(int argc, char *argv[]){
  std::cout<<"Sparse,";
  test_sparse(argc, argv);
  std::cout<<"\n";
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
 omp_set_num_threads(num_threads);
 int use_level_coarsening = 1;
 if(argc >= 4)
  use_level_coarsening = atoi(argv[3]);
 int cp_=10;
 if(argc >= 5)
  cp_ = atoi(argv[4]);
 /// Re-ordering L matrix
#undef METIS
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



 auto *cost = new double[n]();
 for (int i = 0; i < n; ++i) {
  cost[i] = L1_csr->p[i+1] - L1_csr->p[i];
 }
 /// SpTRSV Inspector
 timing_measurement inspect_time; inspect_time.start_timer();
 if(!use_level_coarsening){
  int part_no;
  int lp_=num_threads, ic_ = cp_; bool bp=false;
  //sym_lib::lbc_config(n, L1_csr->p[n], num_threads, lp_, cp_, ic_, bp);
  if(cp_ == 0) cp_ = ic_ = 2;
  get_coarse_levelSet_DAG_CSC_tree(n, L1_csr->p, L1_csr->i,
                                   L1_csr->stype,
                                   final_level_no,
                                   fina_level_ptr,part_no,
                                   final_part_ptr,final_node_ptr,
                                   lp_,cp_, ic_, cost);
 } else{
  schedule_param sp; sp.coarsen_factor = sp.initial_cut = cp_; sp.num_wparts = num_threads;
  build_coarsened_level(n, L1_csc->p, L1_csc->i, num_threads, sp,
                                        coarse_level_no, coarse_level_ptr, coarse_part_ptr, coarse_node_ptr);
 }
 inspect_time.measure_elapsed_time();
 /// SpTRSV Executor
 for (int i = 0; i < NTIMES; ++i) {
  std::copy(y_cpy, y_cpy+n, y);
  timing_measurement t_sptrsv; t_sptrsv.start_timer();
  if(!use_level_coarsening) {
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
 std::cout<<cp_<<",";
 std::cout<<n<<","<<n<<","<<L1_csr->nnz<<","<<t_spmv_sec<<","<<t_sptrsv_sec<<","<<inspect_time.elapsed_time<<",";
 std::cout<<matrix_name<<","<< ( use_level_coarsening ? "LC" : "LBC") <<","<<t_sptrsv_seq_sec<<",";
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
