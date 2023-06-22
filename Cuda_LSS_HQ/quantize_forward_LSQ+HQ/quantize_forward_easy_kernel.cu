#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cmath>
#include <iostream>
#include <sstream>
#include <vector>
#include <cuda.h>
#include <ctime>
#include "cuda_fp16.hpp"
#include "cuda_fp16.h"

#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm_batched.h"
#include "cutlass/layout/matrix.h"

#define N_THREADS 256

template<typename scalar_t>
__global__ void quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * MatO, scalar_t * __restrict__  MatLSQ, const float scale, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        //TODO: Method 2
        // scalar_t temp1 = MatI[x] / scale;
        // int temp2 = temp1;
        // int bias = (temp1 - temp2) * 2;
        // MatO[x] = std::clamp(temp2 + bias, -8, 7);
        float temp = MatI[x] / scale;
        // float tmp = round(MatI[x] / scale);
        MatLSQ[x] = temp;
        MatO[x] = std::clamp((int)(round(temp)), -8, 7);
    }
}

__global__ void pack_cuda_kernel(int8_t * in, int8_t * out, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        out[x] = (in[(x<<1)+1] << 4) | (in[x<<1] & 15);
    }
}

//TODO: Define a CUTLASS GEMM template and launch a GEMM kernel. Int4 gemm
cudaError_t CutlassSgemmNN(
  const int M,
  const int N,
  const int K,
  const cutlass::int4b_t *A,
  int lda,
  const cutlass::int4b_t *B,
  int ldb,
  int32_t *C,
  int ldc) {

// The code section below describes datatype for input, output matrices and computation between
// elements in input matrices.
using ElementAccumulator = int32_t;                 // <- data type of accumulator
using ElementComputeEpilogue = ElementAccumulator;  // <- data type of epilogue operations
using ElementInputA = cutlass::int4b_t;                       // <- data type of elements in input matrix A
using ElementInputB = cutlass::int4b_t;                       // <- data type of elements in input matrix B
using ElementOutput = int32_t;                      // <- data type of elements in output matrix D

// The code section below describes matrix layout of input and output matrices. Column Major for
// Matrix A, Row Major for Matrix B and Row Major for Matrix C
using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::ColumnMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 128>;  // <- threadblock tile M = 128, N = 256, K = 64
// This code section describes tile size a warp will compute
using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;  // <- warp tile M = 64, N = 64, K = 64 
// This code section describes the size of MMA op
using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;  // <- MMA Op tile M = 8, N = 8, K = 16

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;  // <- ??

// This code section describes the epilogue part of the kernel
using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
    ElementOutput,                                     // <- data type of output matrix
    4,  // <- the number of elements per vectorized
                                                       // memory access. For a byte, it's 16
                                                       // elements. This becomes the vector width of
                                                       // math instructions in the epilogue too
    ElementAccumulator,                                // <- data type of accumulator
    ElementComputeEpilogue>;  // <- data type for alpha/beta in linear combination function

// Number of pipelines you want to use
constexpr int NumStages = 3;

using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
                                         LayoutInputA,
                                         ElementInputB,
                                         LayoutInputB,
                                         ElementOutput,
                                         LayoutOutput,
                                         ElementAccumulator,
                                         MMAOp,
                                         SmArch,
                                         ShapeMMAThreadBlock,
                                         ShapeMMAWarp,
                                         ShapeMMAOp,
                                         EpilogueOp,
                                         SwizzleThreadBlock,
                                         NumStages>;
  // Create a tuple of problem size for matrix multiplication
  cutlass::gemm::GemmCoord problem_size(M, N, K);

  // Initialize alpha and beta for dot product computation
  ElementComputeEpilogue alpha = ElementComputeEpilogue(1);
  ElementComputeEpilogue beta = ElementComputeEpilogue(0);
    
  // Split K dimension into 1 partitions
  int split_k_slices = 1;

  // Create a tuple of gemm kernel arguments. This is later passed as arguments to launch
  // instantiated CUTLASS kernel
  typename Gemm::Arguments arguments{problem_size,  // <- problem size of matrix multiplication
                                     {A, lda},  // <- reference to matrix A on device
                                     {B, ldb},  // <- reference to matrix B on device
                                     {C, ldc},  // <- reference to matrix C on device
                                     {C, ldc},  // <- reference to matrix D on device
                                     {alpha, beta},          // <- tuple of alpha and beta
                                     split_k_slices};        // <- k-dimension split factor
  
    Gemm gemm_op;
    cutlass::Status status = gemm_op(arguments);

  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }

  // Return success, if no errors were encountered.
  return cudaSuccess;
}

template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm, scalar_t * __restrict__ output, const float scale, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        output[x] = scale * gemm[x];
        // output[x] = 0;
    }
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, torch::Tensor, long long int> quantize_cuda(torch::Tensor hx, torch::Tensor hy, float scale_x, float scale_y){
    std::vector<double> time_vector;
    cudaError_t result;
    //TODO: remember that input y is transposed
    long long int nx = hx.size(0);
    long long int nz = hx.size(1);
    long long int ny = hy.size(0);

    //TODO: divide hadmard matrix by scale + convert data
    auto option_quantize = torch::TensorOptions().dtype(torch::kInt8).device(hx.device());
    auto option_gemm = torch::TensorOptions().dtype(torch::kInt32).device(hx.device());
    auto option_dequantize = torch::TensorOptions().dtype(hx.dtype()).device(hx.device());
    dim3 block(N_THREADS);

    cudaDeviceSynchronize();
    clock_t time_quantize_start = clock();

    dim3 grid1((nx*nz-1)/(block.x)+1);
    torch::Tensor q_x = torch::empty({nx,nz}, option_quantize);
    torch::Tensor lsq_x = torch::empty({nx,nz}, option_dequantize);
    long long int hx_size = (nx*nz);
    // process of quantize
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hx.scalar_type(), "quantize_cuda", ([&] {
    quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
        hx.data_ptr<scalar_t>(),
        q_x.data_ptr<int8_t>(),
        lsq_x.data_ptr<scalar_t>(),
        scale_x,hx_size);
    }));

    // cudaDeviceSynchronize();
    // clock_t time_quantize1_end = clock();

    dim3 grid2((ny*nz-1)/(block.x)+1);
    torch::Tensor q_y = torch::empty({ny,nz}, option_quantize);
    torch::Tensor lsq_y = torch::empty({ny,nz}, option_dequantize);
    long long int hy_size = (ny*nz);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(hy.scalar_type(), "quantize_cuda", ([&] {
    quantize_cuda_kernel<scalar_t><<<grid2, block>>>(
        hy.data_ptr<scalar_t>(),
        q_y.data_ptr<int8_t>(),
        lsq_y.data_ptr<scalar_t>(),
        scale_y,hy_size);
    }));

    cudaDeviceSynchronize();
    clock_t time_quantize_end = clock();

    //TODO: then pack int8 data into int4 data
    dim3 grid_pack_x((nx*nz/2-1)/block.x+1);
    dim3 grid_pack_y((ny*nz/2-1)/block.x+1);
    torch::Tensor pack_qx = torch::empty({nx,nz>>1}, option_quantize);
    torch::Tensor pack_qy = torch::empty({ny,nz>>1}, option_quantize);
    int qx_size = hx_size >> 1;
    int qy_size = hy_size >> 1;
    pack_cuda_kernel<<<grid_pack_x,block>>>(q_x.data_ptr<int8_t>(), pack_qx.data_ptr<int8_t>(), qx_size);
    //Todo:transpose matrix y to make it colMajor
    // pack_cuda_kernel<<<grid_pack_y,block>>>(q_y.t().contiguous().data_ptr<int8_t>(), pack_qy.data_ptr<int8_t>(), qy_size);
    //Todo:weight needs to be transposed, thus transpose is no longer needed
    pack_cuda_kernel<<<grid_pack_y,block>>>(q_y.data_ptr<int8_t>(), pack_qy.data_ptr<int8_t>(), qy_size);

    cudaDeviceSynchronize();
    clock_t time_pack_end = clock();

    //TODO: then int4 gemm
    int lda = nz;
    int ldb = nz;
    int ldc = ny;
    torch::Tensor gemm = torch::empty({nx, ny}, option_gemm);
    result = CutlassSgemmNN(nx, ny, nz, reinterpret_cast<cutlass::int4b_t *>(pack_qx.data_ptr<int8_t>()), lda, 
            reinterpret_cast<cutlass::int4b_t *>(pack_qy.data_ptr<int8_t>()), ldb, gemm.data_ptr<int32_t>(), ldc);
        
    cudaDeviceSynchronize();
    clock_t time_gemm_end = clock();

    //TODO:Final dequantize
    dim3 grid3((nx*ny-1)/block.x+1);
    torch::Tensor output = torch::empty({nx, ny}, option_dequantize);
    float scale = scale_x * scale_y;
    long long int size = nx * ny;
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(output.scalar_type(), "dequantize_cuda", ([&] {
    dequantize_cuda_kernel<scalar_t><<<grid3, block>>>(
        gemm.data_ptr<int32_t>(), 
        output.data_ptr<scalar_t>(),
        scale,size);
    }));

    cudaDeviceSynchronize();
    clock_t time_dequantize_end = clock();

    // double quantize1_time = (double)(time_quantize1_end - time_quantize1_start) / CLOCKS_PER_SEC;
    double quantize_time = (double)(time_quantize_end - time_quantize_start) / CLOCKS_PER_SEC;
    double pack_time = (double)(time_pack_end - time_quantize_end) / CLOCKS_PER_SEC;
    double gemm_time = (double)(time_gemm_end - time_pack_end) / CLOCKS_PER_SEC;
    double dequantize_time = (double)(time_dequantize_end - time_gemm_end) / CLOCKS_PER_SEC;

    // time_vector.push_back(quantize1_time);
    time_vector.push_back(quantize_time);
    time_vector.push_back(pack_time);
    time_vector.push_back(gemm_time);
    time_vector.push_back(dequantize_time);
 
    // return output;
    return std::make_tuple(output, q_x, q_y, lsq_x, lsq_y, time_vector, gemm, size);
}