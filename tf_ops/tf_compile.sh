#!/usr/bin/env bash

/usr/local/cuda-11.0/bin/nvcc cuda_ulits.cu -o cuda_ulits.cu.o -c -O2 -DGOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

g++ -std=c++11 tf_sampling.cpp tf_gather.cpp tf_interpolate.cpp cuda_ulits.cu.o -o tf_op_so.so -shared -fPIC -I /home/local/CYCLOMEDIA001/qbai/Documents/softwares/miniconda3/envs/GACNet/lib/python3.6/site-packages/tensorflow/include -I /usr/local/cuda-11.0/include -I /home/local/CYCLOMEDIA001/qbai/Documents/softwares/miniconda3/envs/GACNet/lib/python3.6/site-packages/tensorflow/include/external/nsync/public/ -lcudart -L /usr/local/cuda-11.0/lib64/ -L /home/local/CYCLOMEDIA001/qbai/Documents/softwares/miniconda3/envs/GACNet/lib/python3.6/site-packages/tensorflow -L /home/local/CYCLOMEDIA001/qbai/Documents/softwares/miniconda3/envs/GACNet/lib/python3.6/site-packages/tensorflow/lib -L /home/local/CYCLOMEDIA001/qbai/Documents/softwares/miniconda3/envs/GACNet/lib/python3.6/site-packages/tensorflow/ -O2 -D_GLIBCXX_USE_CXX11_ABI=0 /home/local/CYCLOMEDIA001/qbai/Documents/softwares/miniconda3/envs/GACNet/lib/python3.6/site-packages/tensorflow/libtensorflow_framework.so.2



