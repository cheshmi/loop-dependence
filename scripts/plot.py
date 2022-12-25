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
## Update this number
num_cores = 40

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


def plot(csv_path_dense, csv_path_sparse, csv_path_mkls=''):
    df = pd.read_csv(csv_path_dense)
    time_mv = df["MV (sec)"].values
    time_sv = df["SV (sec)"].values
    dim = df["N"].values
    nnz_mv, nnz_sv = df["NNZ"].values, ((df["NNZ"].values - dim)/2)+dim
    flops_mv, flops_sv = (2*nnz_mv)/time_mv/1e9, (2*nnz_sv+dim)/time_sv/1e9


    df_sparse = pd.read_csv(csv_path_sparse)
    spmv_list = filter(df_sparse, Cores=num_cores).sort_values(by=['Matrix Name'], ignore_index=True)
    spmv_list["Speedup"] = spmv_list["MV (sec)"].values / spmv_list["SV (sec)"].values
    mat_names, tuned_row = spmv_list["Matrix Name"].unique(), []
    for mat in mat_names:
        m1 = spmv_list[spmv_list['Matrix Name'] == mat]
        tuned_row.append(spmv_list.iloc[m1["Speedup"].idxmax()])
    tuned_list = pd.concat(tuned_row, ignore_index=True, axis=1).transpose()

    # best_config, config_list = {}, spmv_list.Div.unique()
    # for c in config_list:
    #     c1 = filter(spmv_list, Div=c)
    #     best_config[c] = np.mean(c1["MV (sec)"].values/c1["SV (sec)"].values)
    # res = [key for key in best_config if best_config[key] == max(best_config.values())]
    # tuned_list = filter(spmv_list, Div=res)



    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(20, 12))
    if csv_path_mkls != '':
        fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(37, 12))
    #ax1.plot(nnz, np.ones(len(nnz)), c='black')
    flop_spmv_mkl, flop_sptrsv_mkl = [], []
    if csv_path_mkls != '':
        df_sparse = pd.read_csv(csv_path_mkls)
        spmv_list = filter(df_sparse, Cores=num_cores).sort_values(by=['Matrix Name'])
        nnz_list = spmv_list["NNZ"].values
        flop_spmv_mkl = 2 * nnz_list / spmv_list["MV (sec)"].values / 1e9
        flop_sptrsv_mkl = (2 * nnz_list + spmv_list["N"].values) / spmv_list["SV (sec)"].values / 1e9
        print("Average sparse MKL mv speedup is: ", gmean(flop_spmv_mkl / flop_sptrsv_mkl))

    nnz_list = tuned_list["NNZ"].values
    flop_spmv = 2*nnz_list/tuned_list["MV (sec)"].values/1e9
    # if csv_path_mkls != "": # to have a similar baseline
    #     flop_spmv = flop_spmv_mkl
    flop_sptrsv = (2*nnz_list + tuned_list["N"].values)/tuned_list["SV (sec)"].values/1e9

    max_y = max(max(flop_spmv), max(flop_sptrsv), max(flops_mv), max(flops_sv))
    max_y -= 0.3*max_y

    ax0.scatter(dim, flops_mv, marker='^', c='black', label="Dense Matrix-Vector Multiplication (MKL)")
    ax0.scatter(dim, flops_sv, marker='o', c='red', label="Dense Lower Triangular Solve (MKL)")
    ax1.scatter(nnz_list, flop_spmv, marker='^', c='green', label="Sparse Matrix-Vector Multiplication (Sympiler)")
    ax1.scatter(nnz_list, flop_sptrsv, marker='o', c='m', label="Sparse Lower Triangular Solve (Sympiler)")

    print("Average dense mv speedup is: ", gmean(np.float32(flops_mv/flops_sv)))
    print("Average sparse mv speedup is: ", gmean(np.float32(flop_spmv/flop_sptrsv)))
    if csv_path_mkls != "":
        ax2.scatter(nnz_list, flop_spmv_mkl, marker='^', c='blue', label="Sparse Matrix-Vector Multiplication (MKL)")
        ax2.scatter(nnz_list, flop_sptrsv_mkl, marker='o', c='gray', label="Sparse Lower Triangular Solve (MKL I/E)")
        #max_y = max(max(flop_spmv_mkl), max(flop_sptrsv_mkl))
        ax2.set_ylim(0, max_y)
        ax2.set_yticks(range(0, int(max_y), 2 if num_cores == 4 else 10))
        ax2.set(xlabel="Number of NonZero Elements", ylabel="GFLOP/s", title="(c)")
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.legend(loc='upper left')

    ax1.set_ylim(0, max_y)
    ax0.set_ylim(0, max_y)
    ax0.set_yticks(range(0, int(max_y), 2 if num_cores == 4 else 10))
    ax1.set_yticks(range(0, int(max_y), 2 if num_cores == 4 else 10))
    ax0.set(xlabel="Matrix Dimension", ylabel="GFLOP/s", title="(a)")
    ax1.set(xlabel="Number of NonZero Elements", ylabel="GFLOP/s", title="(b)")
    ax0.spines['top'].set_visible(False)
    ax0.spines['right'].set_visible(False)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax0.legend(loc='upper left')
    ax1.legend(loc='upper left')


    fig.suptitle('Performance of Loop-carried Dependence vs. Parallel Loop, {} cores'.format(num_cores), fontsize=30)
    #plt.show()
    plt.savefig("sympiler_lbc"+str(num_cores)+".png")


if __name__ == '__main__':
    if len(sys.argv) > 3:
        plot(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        plot(sys.argv[1], sys.argv[2])
