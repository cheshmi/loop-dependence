#!/bin/bash


#SBATCH --cpus-per-task=40
#SBATCH --export=ALL
#SBATCH --job-name="TRSV"
#SBATCH --mail-type=begin  # email me when the job starts
#SBATCH --mail-type=end    # email me when the job finishes
#SBATCH --mail-user=kazem.cheshmi@gmail.com
#SBATCH --nodes=1
#SBATCH --output="DDT.%j.%N.out"
#SBATCH -t 12:00:00

BINFILE=./build/loop_CD
SPD_MAT_DIR=${HOME}/UFDB/SPD/
NUM_THREAD=20

MKL_NUM_THREADS=$NUM_THREAD; export MKL_NUM_THREADS
OMP_NUM_THREADS=$NUM_THREAD; export OMP_NUM_THREADS
export MKL_DYNAMIC=FALSE;
export OMP_DYNAMIC=FALSE;
export MKL_VERBOSE=1


mkdir logs
echo "header" > logs/dense.csv
for i in {100..10000..100} ; do
  ${BINFILE} 0 $i $NUM_THREAD >> logs/dense.csv
done


SPD_MATS=($(ls $SPD_MAT_DIR))
echo "header" > logs/sparse_hdag.csv
echo "header" > logs/sparse_lbc.csv
for mat in "${SPD_MATS[@]}"; do
  ${BINFILE} 1 $SPD_MAT_DIR/$mat/$mat.mtx $NUM_THREAD >> logs/sparse_hdag.csv
  ${BINFILE} 1 $SPD_MAT_DIR/$mat/$mat.mtx $NUM_THREAD 1 >> logs/sparse_lbc.csv
done


