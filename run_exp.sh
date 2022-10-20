#!/bin/bash


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="TRSV"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=kazem.cheshmi@gmail.com
#SBATCH --nodes=1
#SBATCH --output="DDT.%j.%N.out"
#SBATCH -t 10:00:00

module load NiaEnv/.2022a
module load intel/2021u4
export MKL_DIR=$MKLROOT
module load cmake
module load gcc


BINFILE=./build/loop_CD
SPD_MAT_DIR=${HOME}/UFDB/SPD/
NUM_THREAD=20

MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;
#export MKL_VERBOSE=1

#### Build
mkdir build
cd build
#make clean
#rm -rf *.txt
cmake -DCMAKE_PREFIX_PATH="$MKLROOT/lib/intel64;$MKLROOT/include;$MKLROOT/../compiler/lib/intel64;_deps/openblas-build/lib/"  -DCMAKE_BUILD_TYPE=Release ..
make -j 20


cd ..


BINDIR=./build/
SPD_MAT_DIR=$HOME/UFDB/SPD/
NUM_THREAD=20

MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;
#export MKL_VERBOSE=1


mkdir logs
echo "Config,N,M,NNZ,MV (sec),SV (sec),Cores,Test," > logs/dense.csv
for i in {100..20000..100} ; do
  ${BINDIR}/dense_test $i $NUM_THREAD >> logs/dense.csv
done


SPD_MATS=($(ls $SPD_MAT_DIR))
echo "Config,N,M,NNZ,MV (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sec),Cores,Test," > logs/sparse_hdag.csv
echo "Config,N,M,NNZ,MV (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sec),Cores,Test," > logs/sparse_lbc.csv
#for mat in "${SPD_MATS[@]}"; do
#  ${BINDIR}/sparse_test $SPD_MAT_DIR/$mat/$mat.mtx $NUM_THREAD >> logs/sparse_hdag.csv
#  ${BINDIR}/sparse_test $SPD_MAT_DIR/$mat/$mat.mtx $NUM_THREAD 1 >> logs/sparse_lbc.csv
#done

for mat in $SPD_MAT_DIR/*.mtx; do
 ${BINDIR}/sparse_test ${mat} $NUM_THREAD >> logs/sparse_hdag.csv
 ${BINDIR}/sparse_test ${mat} $NUM_THREAD 0 >> logs/sparse_lbc.csv
done

echo "Config,N,M,NNZ,MV (sec),MV Inspection (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sec),Cores,Test," > logs/sparse_lbc.csv
for mat in $SPD_MAT_DIR/*.mtx; do
 ${BINDIR}/sparse_mkl_test ${mat} $NUM_THREAD >> logs/sparse_mkl.csv
done