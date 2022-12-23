#!/bin/bash




#### Build
mkdir build
cd build
#make clean
#rm -rf *.txt
/Applications/CLion.app/Contents/bin/cmake/mac/bin/cmake -DCMAKE_PREFIX_PATH="$MKLROOT/lib/intel64;$MKLROOT/include;_deps/openblas-build/lib/"  -DCMAKE_BUILD_TYPE=Release ..
make -j 20


cd ..


BINDIR=./build/
SPD_MAT_DIR=$HOME/UFDB/SPD/
NUM_THREAD=6

MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;
#export MKL_VERBOSE=1


mkdir logs
echo "Config,N,M,NNZ,MV (sec),SV (sec),Cores,Test," > logs/dense.csv
for i in $(seq 100 100 20000); do
  echo ${BINDIR}/dense_test $i $NUM_THREAD >> logs/dense_mac.csv
done


SPD_MATS=($(ls $SPD_MAT_DIR))
echo "Config,Div,N,M,NNZ,MV (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sec),Cores,Test," > logs/sparse_lbc_mac.csv
#for mat in "${SPD_MATS[@]}"; do
#  ${BINDIR}/sparse_test $SPD_MAT_DIR/$mat/$mat.mtx $NUM_THREAD 0 >> logs/sparse_lbc.csv
#done

for mat in $SPD_MAT_DIR/*.mtx; do
  for div in 2 4 6 8 10 20 50 100 200; do
      ${BINDIR}/sparse_test $SPD_MAT_DIR/$mat $NUM_THREAD 0 $div >> logs/sparse_lbc_mac.csv
   done
done

echo "Config,N,M,NNZ,MV (sec),MV Inspection (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sec),Cores,Test," > logs/sparse_mkl_mac.csv
for mat in $SPD_MAT_DIR/*.mtx; do
 ${BINDIR}/sparse_mkl_test ${mat} $NUM_THREAD >> logs/sparse_mkl_mac.csv
done
