# loop-dependence

This repository is a small benchmark to compare how much dependence
between iterations degrade thread parallelism. For details, please 
visit [this blog post](https://blog.cheshmi.cc/loop-carried-dep-vs-parallel.html).

# Prequisites
* CMake and a C++ compiler
* MKL is needed for dense and sparse kernels.
* Sympiler (resolved by cmake)
* Python for data generation and plotting (Numpy, Matplotlib, Pandas)


# Build and Run
First you should clone the repository recursively :
```bash
git clone --recursive  https://github.com/cheshmi/psc_example.git
```

To build and run, run the provided script:
```bash
bash run_niagara.sh
```
The script works on the Niagara server. You may need to change 
modules and paths for a different machines. You will also need
to download sparse matrices that you want to test. 

# Plotting
Once the script is finished, you may use the python script
to plot the data:
```bash
python logs/dense.csv logs/sparse_lbc.csv logs/sparse_mkl.csv
```
The output will be a PNG file in the current directory. 
