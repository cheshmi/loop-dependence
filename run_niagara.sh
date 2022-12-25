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
module load intel/2022u2
export MKL_DIR=$MKLROOT
module load cmake
module load gcc

which cmake
which gcc
which g++
which gdb
which make

#### Build
mkdir build
cd build
#make clean
#rm -rf *.txt
cmake -DCMAKE_PREFIX_PATH="$MKLROOT/lib/intel64;$MKLROOT/include;$MKLROOT/../compiler/lib/intel64;_deps/openblas-build/lib/;/home/m/mmehride/kazem/programs/metis-5.1.0/libmetis;/home/m/mmehride/kazem/programs/metis-5.1.0/include/;"  -DCMAKE_BUILD_TYPE=Release ..
make -j 40


cd ..


BINDIR=./build/
SPD_MAT_DIR=$HOME/UFDB/SPD/
NUM_THREAD=40

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
echo "Config,Div,N,M,NNZ,MV (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sec),Cores,Test," > logs/sparse_lbc.csv
for mat in "${SPD_MATS[@]}"; do
  for div in 2 4 6 8 10 20 50 100 200; do
      ${BINDIR}/sparse_test $SPD_MAT_DIR/$mat $NUM_THREAD 0 $div >> logs/sparse_lbc.csv
   done
done




#echo "Config,N,M,NNZ,MV (sec),MV Inspection (sec),SV (sec),Inspection (sec),Matrix Name,Method,Seq SV (sCores,Test," > logs/sparse_lbc.csv
#for mat in $SPD_MAT_DIR/*.mtx; do
#  ${BINDIR}/sparse_mkl_test ${mat} $NUM_THREAD # >> logs/sparse_mkl.csv
#done
