import sys

import numpy as np
from numpy import ma
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from scipy.stats import gmean

font = {'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)


def filter(df, **kwargs):
    bool_index = None
    for key, value in kwargs.items():
        if isinstance(value, list):
            _bool_index = df[key].isin(value)
        else:
            _bool_index = df[key] == value
        if bool_index is None:
            bool_index = _bool_index
        else:
            bool_index = bool_index & _bool_index
    return df[bool_index]


def plot(csv_path_dense, csv_path_sparse):
    df = pd.read_csv(csv_path_dense)
    time_mv = df["MV (sec)"].values
    time_sv = df["SV (sec)"].values
    dim = df["N"].values
    nnz_mv, nnz_sv = df["NNZ"].values, ((df["NNZ"].values - dim)/2)+dim
    flops_mv, flops_sv = (2*nnz_mv)/time_mv/1e9, (2*nnz_sv+dim)/time_sv/1e9


    df_sparse = pd.read_csv(csv_path_sparse)
    spmv_list = filter(df_sparse, Cores=40).sort_values(by=['Matrix Name'])
    best_config, config_list = {}, spmv_list.Config.unique()
    for c in config_list:
        c1 = filter(spmv_list, Config=c)
        best_config[c] = np.mean(c1["MV (sec)"].values/c1["SV (sec)"].values)
    res = [key for key in best_config if best_config[key] == max(best_config.values())]
    tuned_list = filter(spmv_list, Config=res)
    nnz_list = tuned_list["NNZ"].values
    flop_spmv = 2*nnz_list/tuned_list["MV (sec)"].values/1e9
    flop_sptrsv = (2*nnz_list + tuned_list["N"].values)/tuned_list["SV (sec)"].values/1e9



    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20,12))
    ax0.scatter(dim, flops_mv, marker='^', c='black', label="Dense Matrix-Vector Multiplication")
    ax0.scatter(dim, flops_sv, marker='o', c='red', label="Dense Lower Triangular Solve")

    ax1.scatter(nnz_list, flop_spmv, marker='^', c='blue', label="Sparse Matrix-Vector Multiplication")
    ax1.scatter(nnz_list, flop_sptrsv, marker='o', c='m', label="Sparse Lower Triangular Solve")

    print("Average dense mv speedup is: ", gmean(flops_mv/flops_sv))
    print("Average sparse mv speedup is: ", gmean(flop_spmv/flop_sptrsv))

    #ax1.plot(nnz, np.ones(len(nnz)), c='black')
    max_y = max(max(flop_spmv), max(flop_sptrsv), max(flops_mv), max(flops_sv))
    ax1.set_ylim(0, max_y)
    ax0.set_ylim(0, max_y)
    ax0.set_yticks(range(0, int(max_y), 10))
    ax1.set_yticks(range(0, int(max_y), 10))
    ax0.set(xlabel="Matrix Dimension", ylabel="GFLOP/s")
    ax1.set(xlabel="Number of NonZero Elements", ylabel="GFLOP/s")
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax0.legend(loc='upper left')
    ax1.legend(loc='upper left')
    fig.suptitle('Performance of Loop-carried Dependence vs. Parallel Loop, Sparse and Dense', fontsize=30)
    #plt.show()
    plt.savefig("sympiler_lbc.png")


if __name__ == '__main__':
    plot(sys.argv[1], sys.argv[2])
