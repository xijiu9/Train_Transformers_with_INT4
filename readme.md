Cuda_LSS_HQ provides the CUDA implementation of our algorithm. SingleDivide provides the training recipe to reproduce the result.

To reproduce the result, please first change the cutlass_path in each of the setup.py in Cuda_LSS_HQ to the absolute path of cutlass/include.

Then, please run 
```
bash task.sh
```
in SingleDivide reproduce the result of GLUE dataset.