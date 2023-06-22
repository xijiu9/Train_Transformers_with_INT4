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
#include <curand_kernel.h>
#include <tuple>
#include <bits/stdc++.h>
// #include <torch/distributions/gumbel.h>

#include "torch/script.h"
using namespace torch::indexing;

template<typename scalar_t>
__global__ void quantize_cuda_kernel(const scalar_t * __restrict__  MatI, int8_t * first_transform, int8_t * second_transform, 
                                    const int num_bins_half, const int num_bins_clamp, const float scale, long long int size, unsigned long seed){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        // set random value
        curandStatePhilox4_32_10_t state;
        curand_init(seed, x, 0, &state);
        const float noise = curand_uniform(&state);

        float trans_input = MatI[x] * scale;

        float tmp1 = round(trans_input / num_bins_half);
        int firstTransform = std::clamp((int)(tmp1), -num_bins_clamp, num_bins_clamp);
        first_transform[x] = firstTransform;
        // float quantize = (transform + 8) / scale + zero_point;
        // first_quantize[x] = firstTransform * num_bins_half / scale;

        float tmp2 = round(trans_input - firstTransform * num_bins_half + noise - 0.5);
        int secondTransform = std::clamp((int)(tmp2), -num_bins_clamp, num_bins_clamp);
        second_transform[x] = secondTransform;
        // second_quantize[x] = secondTransform / scale;
    }
}

__global__ void pack_cuda_kernel(int8_t * in, int8_t * out, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    if (x<size){
        out[x] = (in[(x<<1)+1] << 4) | (in[x<<1] & 15);
    }
}

template<typename scalar_t>
__global__ void multiple_kernel(const scalar_t * __restrict__ in, scalar_t * __restrict__ out, float scale, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x<size){
        out[x] = in[x] * scale;
    }
}

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
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

#define N_THREADS 256

template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm1, const int32_t * gemm2, const scalar_t * __restrict__ gemm3, 
                                        const scalar_t * __restrict__ gemm4, scalar_t * __restrict__ output, 
                                        const float scale_gemm1, const float scale_gemm2, long long int size){  
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    // int row = x / ny, col = x - row * ny;

    if (x<size){
       output[x] = gemm1[x] * scale_gemm1 + gemm2[x] * scale_gemm2 + gemm3[x] + gemm4[x];
        // output[x] = 0;
    }
}

template<typename scalar_t>
__global__ void dequantize2_cuda_kernel(const int32_t * gemm1, const int32_t * gemm2, scalar_t * __restrict__ output, 
                                        const float scale_gemm1, const float scale_gemm2, long long int size){  
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x<size){
       output[x] = gemm1[x] * scale_gemm1 + gemm2[x] * scale_gemm2;
    }
}

template<typename scalar_t>
__global__ void LSQ_cuda_kernel(const scalar_t * lsq_weight, const scalar_t * __restrict__ grad_output, scalar_t * __restrict__ grad_alpha_out, 
                                scalar_t * __restrict__ grad_input, const float grad_scale, const long long int size){  
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x<size){
       scalar_t q_w = lsq_weight[x];
       scalar_t indicate_small = (q_w < -8);
       scalar_t indicate_big = (q_w > 7);
       scalar_t indicate_middle = 1.0 - indicate_small - indicate_big;
       scalar_t grad_out = grad_output[x];
       grad_alpha_out[x] = (indicate_small * -8 + indicate_big * 7 + indicate_middle * (-q_w + round(q_w))) * grad_out * grad_scale;
       grad_input[x] = indicate_middle * grad_out;
    }
}


__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
}

__device__ __inline__ c10::Half __shfl_sync(const unsigned mask, const c10::Half var,
                                            const int delta, const int width) {
  __half var_ = var;
  return __shfl_sync(mask, var_, delta, width);
}

template <typename scalar_t>
__global__ void minimax_cuda_kernel(const scalar_t* __restrict__ data,
                                    scalar_t* __restrict__ min,
                                    scalar_t* __restrict__ max,
                                    int64_t N,
                                    int64_t D) {
  scalar_t max_val, min_val;
  max_val = -1e30;
  min_val = 1e30;

  for (int64_t k1_outer = 0; k1_outer < D / 32; ++k1_outer) {
    max_val = std::max(max_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
    min_val = std::min(min_val, data[blockIdx.x * D + k1_outer * 32 + threadIdx.x]);
  }

  unsigned int mask;
  scalar_t max_val_t, min_val_t;
  mask = __activemask();

  max_val_t = __shfl_down_sync(mask, max_val, 16, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 8, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 4, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 2, 32);
  max_val = std::max(max_val, max_val_t);
  max_val_t = __shfl_down_sync(mask, max_val, 1, 32);
  max_val = std::max(max_val, max_val_t);
  max_val = __shfl_sync(mask, max_val, 0, 32);
  max[blockIdx.x] = max_val;

  min_val_t = __shfl_down_sync(mask, min_val, 16, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 8, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 4, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 2, 32);
  min_val = std::min(min_val, min_val_t);
  min_val_t = __shfl_down_sync(mask, min_val, 1, 32);
  min_val = std::min(min_val, min_val_t);
  min_val = __shfl_sync(mask, min_val, 0, 32);
  min[blockIdx.x] = min_val;
}

//TODO: N means rows, D means cols
template<typename scalar_t>
__global__ void linalg_norm_cuda_kernel(const scalar_t * __restrict__ in, float * linalg, int N, int D, int stride_D){
  float sum_val = 0;

  for (int64_t k1_outer = 0; k1_outer < stride_D; ++k1_outer) {
    float temp = in[blockIdx.x * D + (k1_outer << 5) + threadIdx.x];
    sum_val += temp * temp;
  }

  unsigned int mask;
  float sum_val_t;
  mask = __activemask();

  sum_val_t = __shfl_down_sync(mask, sum_val, 16, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 8, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 4, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 2, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 1, 32);
  sum_val += sum_val_t;
  linalg[blockIdx.x] = sqrt(sum_val);
}

__global__ void linalg_normInt_cuda_kernel(const int8_t * in, float * linalg, int N, int D, int stride_D, float scale){
  float sum_val = 0;

  for (int64_t k1_outer = 0; k1_outer < stride_D; ++k1_outer) {
    int64_t temp = in[blockIdx.x * D + (k1_outer << 5) + threadIdx.x];
    sum_val += temp * temp;
  }

  unsigned int mask;
  float sum_val_t;
  mask = __activemask();

  sum_val_t = __shfl_down_sync(mask, sum_val, 16, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 8, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 4, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 2, 32);
  sum_val += sum_val_t;
  sum_val_t = __shfl_down_sync(mask, sum_val, 1, 32);
  sum_val += sum_val_t;
  linalg[blockIdx.x] = sqrt(sum_val) * scale;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, float> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_weight){
    std::vector<double> time_vector;
    long long int nz = x.size(0);
    long long int nx = x.size(1);
    long long int ny = qy.size(1);

    cudaDeviceSynchronize();
    clock_t time_quantize_start = clock();

    auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto option_quantize = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto option_float = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    torch::Tensor first_transform = torch::empty({nz, nx}, option_transform);
    // torch::Tensor first_quantize = torch::empty({nz, nx}, option_quantize);
    torch::Tensor second_transform = torch::empty({nz, nx}, option_transform);
    // torch::Tensor second_quantize = torch::empty({nz, nx}, option_quantize);

    dim3 block(N_THREADS);
    dim3 grid1((nx*nz-1)/block.x+1);
    long long int size_quantize = nz * nx ;
    // process of first quantize
    torch::Tensor min_x = torch::empty({nz, }, option_quantize);
    torch::Tensor max_x = torch::empty({nz, }, option_quantize);
    int minimax_blocks = nz;
    int minimax_threads = 32;

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "minimax_cuda", ([&] {
    minimax_cuda_kernel<scalar_t><<<minimax_blocks, minimax_threads>>>(
      x.data_ptr<scalar_t>(), min_x.data_ptr<scalar_t>(), max_x.data_ptr<scalar_t>(),
      nz, nx);
    }));
    // float mn = std::min(x.min().item<float>() - 1e-8, 0.);
    // float mx = std::max(x.max().item<float>() + 1e-8, 0.);
    float mn = std::min(min_x.min().item<float>() - 1e-8, 0.);
    float mx = std::max(max_x.max().item<float>() + 1e-8, 0.);

    int num_bins_half = pow(2, num_bits) - 2;
    int num_bins = num_bins_half * num_bins_half;
    int num_bins_clamp = num_bins_half / 2 - 1;

    float scale1 = num_bins / (2 * max(fabs(mn), fabs(mx)));

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "quantize_cuda", ([&] {
    quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
        x.data_ptr<scalar_t>(),
        first_transform.data_ptr<int8_t>(),
        // first_quantize.data_ptr<scalar_t>(),
        second_transform.data_ptr<int8_t>(),
        // second_quantize.data_ptr<scalar_t>(),
        num_bins_half, num_bins_clamp,
        scale1, size_quantize,rand());
    }));

    cudaDeviceSynchronize();
    clock_t time_quantize_end = clock();

    // leverage score
    // TODO: use dim=0 because torch.linalg only supports dim=1
    int threads = 32;
    int blocks = nz;

    auto x1_len = torch::empty({nz,}, option_float);
    auto x2_len = torch::empty({nz,}, option_float);
    auto y_len = torch::empty({nz,}, option_float);

    int stride_x = nx / 32;
    float scale_x1 = num_bins_half / scale1;
    float scale_x2 = 1. / scale1;
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(first_quantize.scalar_type(), "linalg_cuda", ([&] {
    // linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
    //     first_quantize.data_ptr<scalar_t>(), 
    //     x1_len.data_ptr<float>(),
    //     nz,nx,stride_x);
    // }));
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(second_quantize.scalar_type(), "linalg_cuda", ([&] {
    // linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
    //     second_quantize.data_ptr<scalar_t>(), 
    //     x2_len.data_ptr<float>(),
    //     nz,nx,stride_x);
    // }));

    linalg_normInt_cuda_kernel<<<blocks, threads>>>(
        first_transform.data_ptr<int8_t>(), 
        x1_len.data_ptr<float>(),
        nz,nx,stride_x, scale_x1);

    linalg_normInt_cuda_kernel<<<blocks, threads>>>(
        second_transform.data_ptr<int8_t>(), 
        x2_len.data_ptr<float>(),
        nz,nx,stride_x, scale_x2);

    int stride_y = ny / 32;
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(y.scalar_type(), "linalg_cuda", ([&] {
    // linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
    //     y.data_ptr<scalar_t>(), 
    //     y_len.data_ptr<float>(),
    //     nz,ny,stride_y);
    // }));
    linalg_normInt_cuda_kernel<<<blocks, threads>>>(
        qy.data_ptr<int8_t>(), 
        y_len.data_ptr<float>(),
        nz,ny,stride_y, scaley);

    // TODO: whether need to change dtype from half into float? It depends 
    auto vec_norm = torch::cat({torch::mul(x1_len, y_len), torch::mul(x2_len, y_len)});
    // auto vec_norm = torch::cat({torch::mul(x1_len, y_len), torch::mul(x2_len, y_len)});
    int len_norm = vec_norm.numel();

    cudaDeviceSynchronize();
    clock_t time_leverage_end = clock();

    int cnt = 0;
    int flag = 0;
    // auto norm_weight_loop = vec_norm * len_norm / (2 * vec_norm.sum());
    auto norm_weight_loop = torch::empty_like(vec_norm);
    float scale_norm = len_norm / (2 * vec_norm.sum().item<float>());
    dim3 grid_norm(len_norm/block.x+1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vec_norm.scalar_type(), "multiple_cuda", ([&] {
    multiple_kernel<scalar_t><<<grid_norm, block>>>(
        vec_norm.data_ptr<scalar_t>(),
        norm_weight_loop.data_ptr<scalar_t>(),
        scale_norm,len_norm);
    }));
    auto sample_index = torch::empty_like(norm_weight_loop);
    int posNum = (norm_weight_loop > 0).sum().item<int>();
    // if (posNum < len_norm / 2){
    if (true) {
    // if (false) {
        cnt = posNum;
        norm_weight_loop.index_put_({norm_weight_loop > 0}, 1);
        sample_index = norm_weight_loop;
        flag = 2;
    }else{
        bool whileloop = (norm_weight_loop.max() > 1).item<bool>();
        while (1){
            if (!(whileloop && cnt < len_norm / 2)) {
                flag = 1;
                break;
            }
            auto small_index = (norm_weight_loop < 1);
            auto small_value = norm_weight_loop.index({small_index});
            long long int small_len = small_value.numel();
            cnt = len_norm - small_len;
            norm_weight_loop = torch::clamp(norm_weight_loop, 0, 1);
            bool breakloop = (small_value.max() == 0).item<bool>();
            if (breakloop) {
                flag = 2;
                break;
            }
            // small_value = small_value * (len_norm / 2 - cnt) / small_value.sum();
            float scale_small = (len_norm / 2 - cnt) / small_value.sum().item<float>();
            dim3 grid_small(small_len/block.x+1);
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(small_value.scalar_type(), "multiple_cuda", ([&] {
            multiple_kernel<scalar_t><<<grid_small, block>>>(
                small_value.data_ptr<scalar_t>(),
                small_value.data_ptr<scalar_t>(),
                scale_small,small_len);
            }));
            // norm_weight_loop[small_index] = small_value;
            norm_weight_loop.index_put_({small_index}, small_value);
            whileloop = (norm_weight_loop.max() > 1).item<bool>();
        } 
        sample_index = torch::bernoulli(norm_weight_loop);
    }
    // auto sample_index = torch::bernoulli(norm_weight_loop);
    auto small_indices = torch::nonzero(sample_index.index({Slice({None, len_norm/2})}) == 1).squeeze(1);
    auto large_indices = torch::nonzero(sample_index.index({Slice(len_norm/2)}) == 1).squeeze(1);

    auto option_output = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto grad_output = torch::empty({nx,ny}, option_output);
    dim3 grid2((nx*ny-1)/block.x+1);
    long long int size = nx*ny;

    clock_t time_sampleflag_end;
    clock_t time_sample_end;
    clock_t time_pack_end;
    clock_t time_gemm_end;
    clock_t time_dequantize_end;
    cudaDeviceSynchronize();
    clock_t time_method_start = clock();
    int small_num_, large_num_;

    if (flag == 1){
        auto norm_small_indices = (norm_weight_loop.index({small_indices}) == 1);
        auto small_num = norm_small_indices.sum();
        // TODO: test if .int() can work in libtorch
        small_num = ((small_num / 32).floor() * 32).to(torch::kInt32);
        small_num_ = small_num.item<int>();
        auto small_int_indices = small_indices.index({norm_small_indices}).index({Slice({None, small_num_})});
        auto small_left_indices = small_indices.index({~torch::isin(small_indices, small_int_indices)});
        auto norm_large_indices = (norm_weight_loop.index({large_indices + len_norm / 2}) == 1);
        auto large_num = norm_large_indices.sum();
        // TODO: test if .int() can work in libtorch
        large_num = ((large_num / 32).floor() * 32).to(torch::kInt32);
        large_num_ = large_num.item<int>();
        auto large_int_indices = large_indices.index({norm_large_indices}).index({Slice({None, large_num_})});
        auto large_left_indices = large_indices.index({~torch::isin(large_indices, large_int_indices)});
        // auto _index = torch::nonzero((sample_index == 1)).squeeze();

        norm_weight_loop.index_put_({norm_weight_loop == 0}, 1e-10);

        
        // auto small_num_ = (_index < len_norm / 2).sum();
        // auto large_num_ = _index.numel() - small_num_;
        // auto small_indices = _index.index({Slice({None, small_num_.item<int>()})});
        // auto large_indices = _index.index({Slice(small_num_.item<int>())}) - int(len_norm / 2);
        // auto norm_weight_small = norm_weight_loop.index({small_indices});
        // auto norm_weight_large = norm_weight_loop.index({large_indices + len_norm / 2});
        // auto output = torch::cat({first_quantize, second_quantize});
        // output = output / norm_weight_loop.unsqueeze(1);
        cudaDeviceSynchronize();
        time_sampleflag_end = clock();


        //TODO: suppose an easy situation so that it can be faster
        auto sample_x1 = first_transform.index({small_int_indices}).t().contiguous();
        auto sample_x2 = second_transform.index({large_int_indices}).t().contiguous();
        auto sample_y1 = qy.index({small_int_indices}).t().contiguous();
        auto sample_y2 = qy.index({large_int_indices}).t().contiguous();
        auto sample_x3 = (first_transform.index({small_left_indices}).t() * num_bins_half / (scale1 * norm_weight_loop.index({small_left_indices}))).to(x.dtype());
        auto sample_x4 = (second_transform.index({large_left_indices}).t() / (scale1 * norm_weight_loop.index({large_left_indices + len_norm / 2}))).to(x.dtype());
        //todo:currently multiply a scaley to convert it into fp16
        auto sample_y3 = (qy.index({small_left_indices}) * scaley).to(x.dtype());
        auto sample_y4 = (qy.index({large_left_indices}) * scaley).to(x.dtype());

        cudaDeviceSynchronize();
        time_sample_end = clock();

        // pack process
        // auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
        auto sample_x1_int4 = torch::empty({nx, small_num_>>1}, option_transform);
        auto sample_x2_int4 = torch::empty({nx, large_num_>>1}, option_transform);
        auto sample_y1_int4 = torch::empty({ny, small_num_>>1}, option_transform);
        auto sample_y2_int4 = torch::empty({ny, large_num_>>1}, option_transform);
        long long int grid_size_x1 = nx*small_num_/2;
        long long int grid_size_x2 = nx*large_num_/2;
        long long int grid_size_y1 = ny*small_num_/2;
        long long int grid_size_y2 = ny*large_num_/2;
        dim3 grid_pack_x1((grid_size_x1-1)/block.x+1);
        dim3 grid_pack_x2((grid_size_x2-1)/block.x+1);
        dim3 grid_pack_y1((grid_size_y1-1)/block.x+1);
        dim3 grid_pack_y2((grid_size_y2-1)/block.x+1);
        if (small_num_ > 0) {
            pack_cuda_kernel<<<grid_pack_x1,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x1);
            pack_cuda_kernel<<<grid_pack_y1,block>>>(sample_y1.data_ptr<int8_t>(), sample_y1_int4.data_ptr<int8_t>(), grid_size_y1);
        }
        if (large_num_ > 0) {
            pack_cuda_kernel<<<grid_pack_x2,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x2);
            pack_cuda_kernel<<<grid_pack_y2,block>>>(sample_y2.data_ptr<int8_t>(), sample_y2_int4.data_ptr<int8_t>(), grid_size_y2);
        }

        cudaDeviceSynchronize();
        time_pack_end = clock();

        // gemm process
        cudaError_t result;
        int lda_first = small_num_;
        int ldb_first = small_num_;
        int ldc = ny;
        // Chunked matrix multiplication
        auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
        if (small_num_ > 0) {
            result = CutlassSgemmNN(nx, ny, small_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda_first, reinterpret_cast<cutlass::int4b_t *>(sample_y1_int4.data_ptr<int8_t>()), ldb_first, gemm1.data_ptr<int32_t>(), ldc);
        } else {
            gemm1 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
        }
        // result = CutlassSgemmNN(nx, ny, small_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda_first, reinterpret_cast<cutlass::int4b_t *>(sample_y1_int4.data_ptr<int8_t>()), ldb_first, gemm1.data_ptr<int32_t>(), ldc);

        int lda_second = large_num_;
        int ldb_second = large_num_;
        auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
        if (large_num_ > 0) {
            result = CutlassSgemmNN(nx, ny, large_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda_second, reinterpret_cast<cutlass::int4b_t *>(sample_y2_int4.data_ptr<int8_t>()), ldb_second, gemm2.data_ptr<int32_t>(), ldc);
        } else {
            gemm2 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
        }
        // result = CutlassSgemmNN(nx, ny, large_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda_second, reinterpret_cast<cutlass::int4b_t *>(sample_y2_int4.data_ptr<int8_t>()), ldb_second, gemm2.data_ptr<int32_t>(), ldc);

        // cudaDeviceSynchronize();
        // clock_t time_int4gemm_end = clock();

        auto gemm3 = torch::matmul(sample_x3, sample_y3);
        auto gemm4 = torch::matmul(sample_x4, sample_y4);
        // auto gemm3 = torch::empty({nx, ny}, at::device(at::kCUDA).dtype(torch::kFloat16));
        // auto gemm4 = torch::empty({nx, ny}, at::device(at::kCUDA).dtype(torch::kFloat16));


        // auto gemm3 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kFloat16));

        // auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kFloat16));

        // cudaDeviceSynchronize();
        // clock_t time_fp16gemm_end = clock();
        cudaDeviceSynchronize();
        time_gemm_end = clock();

        // dequantize process
        // First dequantize higher 4 bits
        // auto sum_y1_column = torch::sum(qy.index({small_int_indices}), 0);
        // auto sum_y2_column = torch::sum(qy.index({large_int_indices}), 0);

        // float const_x1 = (8.0 / scale1 + zero_point1) * scaley;
        // float const_x2 = (8.0 / scale2 + zero_point2) * scaley;
        // float const_x = const_x1 + const_x2;
        float scale_gemm1 = scaley * num_bins_half / (scale1);
        float scale_gemm2 = scaley / (scale1);
        AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize_cuda", ([&] {
        dequantize_cuda_kernel<scalar_t><<<grid2, block>>>(
            gemm1.data_ptr<int32_t>(), 
            gemm2.data_ptr<int32_t>(),
            gemm3.data_ptr<scalar_t>(), 
            gemm4.data_ptr<scalar_t>(),
            grad_output.data_ptr<scalar_t>(),
            // sum_y1_column.data_ptr<int64_t>(),
            // sum_y2_column.data_ptr<int64_t>(),
            // const_x1, const_x2, 
            scale_gemm1, scale_gemm2,
            size);
        }));

        cudaDeviceSynchronize();
        time_dequantize_end = clock();

    } else if (flag == 2){
        // if (cnt > 800){
        if (false) {
            small_num_ = floor(small_indices.numel() / 32.0) * 32;
            large_num_ = floor(large_indices.numel() / 32.0) * 32;

            auto small_int_indices = small_indices.index({Slice({None, small_num_})});
            auto large_int_indices = large_indices.index({Slice({None, large_num_})});
            cudaDeviceSynchronize();
            time_sampleflag_end = clock();

            auto sample_x1 = first_transform.index({small_int_indices}).t().contiguous();
            auto sample_x2 = second_transform.index({large_int_indices}).t().contiguous();
            auto sample_y1 = qy.index({small_int_indices}).t().contiguous();
            auto sample_y2 = qy.index({large_int_indices}).t().contiguous();

            cudaDeviceSynchronize();
            time_sample_end = clock();

            auto sample_x1_int4 = torch::empty({nx, small_num_>>1}, option_transform);
            auto sample_x2_int4 = torch::empty({nx, large_num_>>1}, option_transform);
            auto sample_y1_int4 = torch::empty({ny, small_num_>>1}, option_transform);
            auto sample_y2_int4 = torch::empty({ny, large_num_>>1}, option_transform);
            long long int grid_size_x1 = nx*small_num_/2;
            long long int grid_size_x2 = nx*large_num_/2;
            long long int grid_size_y1 = ny*small_num_/2;
            long long int grid_size_y2 = ny*large_num_/2;
            dim3 grid_pack_x1((grid_size_x1-1)/block.x+1);
            dim3 grid_pack_x2((grid_size_x2-1)/block.x+1);
            dim3 grid_pack_y1((grid_size_y1-1)/block.x+1);
            dim3 grid_pack_y2((grid_size_y2-1)/block.x+1);
            if (small_num_ > 0) {
                pack_cuda_kernel<<<grid_pack_x1,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x1);
                pack_cuda_kernel<<<grid_pack_y1,block>>>(sample_y1.data_ptr<int8_t>(), sample_y1_int4.data_ptr<int8_t>(), grid_size_y1);
            }
            if (large_num_ > 0) {
                pack_cuda_kernel<<<grid_pack_x2,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x2);
                pack_cuda_kernel<<<grid_pack_y2,block>>>(sample_y2.data_ptr<int8_t>(), sample_y2_int4.data_ptr<int8_t>(), grid_size_y2);
            }
            cudaDeviceSynchronize();
            time_pack_end = clock();

            cudaError_t result;
            int lda_first = small_num_;
            int ldb_first = small_num_;
            int ldc = ny;
            // Chunked matrix multiplication
            auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            if (small_num_ > 0) {
                result = CutlassSgemmNN(nx, ny, small_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda_first, reinterpret_cast<cutlass::int4b_t *>(sample_y1_int4.data_ptr<int8_t>()), ldb_first, gemm1.data_ptr<int32_t>(), ldc);
            } else {
                gemm1 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            }

            int lda_second = large_num_;
            int ldb_second = large_num_;
            auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            if (large_num_ > 0) {
                result = CutlassSgemmNN(nx, ny, large_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda_second, reinterpret_cast<cutlass::int4b_t *>(sample_y2_int4.data_ptr<int8_t>()), ldb_second, gemm2.data_ptr<int32_t>(), ldc);
            } else {
                gemm2 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            }
            cudaDeviceSynchronize();
            time_gemm_end = clock();

            // First dequantize higher 4 bits
            float scale_gemm1 = scaley * num_bins_half / (scale1);
            float scale_gemm2 = scaley / (scale1);
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize2_cuda", ([&] {
            dequantize2_cuda_kernel<scalar_t><<<grid2, block>>>(
                gemm1.data_ptr<int32_t>(), 
                gemm2.data_ptr<int32_t>(),
                grad_output.data_ptr<scalar_t>(),
                scale_gemm1, scale_gemm2,
                size);
            }));
            cudaDeviceSynchronize();
            time_dequantize_end = clock();
        } else{
            small_num_ = ceil(small_indices.numel() / 32.0) * 32;
            large_num_ = ceil(large_indices.numel() / 32.0) * 32;

            // int small_pad = small_num_ - small_indices.numel();
            // if (small_pad > 0) {
            //     auto small_pad_indices = (torch::nonzero(sample_index.index({Slice({None, len_norm/2})}) != 1).squeeze(1)).index({Slice({None, small_pad})});
            //     if (small_pad_indices.numel() < small_pad) exit(0);
            //     small_indices = torch::cat({small_indices, small_pad_indices}, 0);
            // }
            // int large_pad = large_num_ - large_indices.numel();
            // if (large_pad > 0) {
            //     auto large_pad_indices = (torch::nonzero(sample_index.index({Slice(len_norm/2)}) != 1).squeeze(1)).index({Slice({None, large_pad})}); 
            //     if (large_pad_indices.numel() < large_pad) exit(0);
            //     large_indices = torch::cat({large_indices, large_pad_indices}, 0);
            // }
            

            // // std::vector<int64_t> padding_small = {0,0,0,small_num_ - small_indices.numel()};
            // // std::vector<int64_t> padding_large = {0,0,0,large_num_ - large_indices.numel()};

            // // torch::nn::ZeroPad2d pad_small(padding_small);
            // // torch::nn::ZeroPad2d pad_large(padding_large);
            // cudaDeviceSynchronize();
            // time_sampleflag_end = clock();

            // auto sample_x1 = first_transform.index({small_indices}).t().contiguous();
            // auto sample_x2 = second_transform.index({large_indices}).t().contiguous();
            // auto sample_y1 = qy.index({small_indices}).t().contiguous();
            // auto sample_y2 = qy.index({large_indices}).t().contiguous();

            std::vector<int64_t> padding_small = {0,0,0,small_num_ - small_indices.numel()};
            std::vector<int64_t> padding_large = {0,0,0,large_num_ - large_indices.numel()};

            torch::nn::ZeroPad2d pad_small(padding_small);
            torch::nn::ZeroPad2d pad_large(padding_large);
            cudaDeviceSynchronize();
            time_sampleflag_end = clock();

            auto sample_x1 = pad_small(first_transform.index({small_indices})).t().contiguous();
            auto sample_x2 = pad_large(second_transform.index({large_indices})).t().contiguous();
            auto sample_y1 = pad_small(qy.index({small_indices})).t().contiguous();
            auto sample_y2 = pad_large(qy.index({large_indices})).t().contiguous();

            cudaDeviceSynchronize();
            time_sample_end = clock();

            auto sample_x1_int4 = torch::empty({nx, small_num_>>1}, option_transform);
            auto sample_x2_int4 = torch::empty({nx, large_num_>>1}, option_transform);
            auto sample_y1_int4 = torch::empty({ny, small_num_>>1}, option_transform);
            auto sample_y2_int4 = torch::empty({ny, large_num_>>1}, option_transform);
            long long int grid_size_x1 = nx*small_num_/2;
            long long int grid_size_x2 = nx*large_num_/2;
            long long int grid_size_y1 = ny*small_num_/2;
            long long int grid_size_y2 = ny*large_num_/2;
            dim3 grid_pack_x1((grid_size_x1-1)/block.x+1);
            dim3 grid_pack_x2((grid_size_x2-1)/block.x+1);
            dim3 grid_pack_y1((grid_size_y1-1)/block.x+1);
            dim3 grid_pack_y2((grid_size_y2-1)/block.x+1);
            if (small_num_ > 0) {
                pack_cuda_kernel<<<grid_pack_x1,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x1);
                pack_cuda_kernel<<<grid_pack_y1,block>>>(sample_y1.data_ptr<int8_t>(), sample_y1_int4.data_ptr<int8_t>(), grid_size_y1);
            }
            if (large_num_ > 0) {
                pack_cuda_kernel<<<grid_pack_x2,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x2);
                pack_cuda_kernel<<<grid_pack_y2,block>>>(sample_y2.data_ptr<int8_t>(), sample_y2_int4.data_ptr<int8_t>(), grid_size_y2);
            }

            cudaDeviceSynchronize();
            time_pack_end = clock();

            cudaError_t result;
            int lda_first = small_num_;
            int ldb_first = small_num_;
            int ldc = ny;
            // Chunked matrix multiplication
            auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            if (small_num_ > 0) {
                result = CutlassSgemmNN(nx, ny, small_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda_first, reinterpret_cast<cutlass::int4b_t *>(sample_y1_int4.data_ptr<int8_t>()), ldb_first, gemm1.data_ptr<int32_t>(), ldc);
            } else {
                gemm1 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            }

            int lda_second = large_num_;
            int ldb_second = large_num_;
            auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            if (large_num_ > 0) {
                result = CutlassSgemmNN(nx, ny, large_num_, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda_second, reinterpret_cast<cutlass::int4b_t *>(sample_y2_int4.data_ptr<int8_t>()), ldb_second, gemm2.data_ptr<int32_t>(), ldc);
            } else {
                gemm2 = torch::zeros({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
            }
            cudaDeviceSynchronize();
            time_gemm_end = clock();

            // First dequantize higher 4 bits
            float scale_gemm1 = scaley * num_bins_half / (scale1);
            float scale_gemm2 = scaley / (scale1);
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize2_cuda", ([&] {
            dequantize2_cuda_kernel<scalar_t><<<grid2, block>>>(
                gemm1.data_ptr<int32_t>(), 
                gemm2.data_ptr<int32_t>(),
                grad_output.data_ptr<scalar_t>(),
                scale_gemm1, scale_gemm2,
                size);
            }));
            cudaDeviceSynchronize();
            time_dequantize_end = clock();
        }
        
    }

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize_cuda", ([&] {
    // dequantize_cuda_kernel_fp16<scalar_t><<<grid2, block>>>(
    //     gemm1.data_ptr<scalar_t>(), 
    //     gemm2.data_ptr<scalar_t>(),
    //     grad_output.data_ptr<scalar_t>(),
    //     size);
    // }));

    float grad_scale = 1.0 / sqrt(lsq_weight.numel() * 7);
    auto grad_alpha_out = torch::empty({nx,ny}, option_output);
    auto grad_input = torch::empty({nx,ny}, option_output);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "LSQ_cuda", ([&] {
    LSQ_cuda_kernel<scalar_t><<<grid2, block>>>(
        lsq_weight.data_ptr<scalar_t>(), 
        grad_output.data_ptr<scalar_t>(),
        grad_alpha_out.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(),
        grad_scale, size);
    }));

    // auto q_w = hadamard_weight / scale_weight;
    // auto indicate_small = (q_w < -8).to(torch::kFloat16);
    // auto indicate_big = (q_w > 7).to(torch::kFloat16);
    // auto indicate_middle = 1.0 - indicate_small - indicate_big;
    // auto grad_alpha = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
    //                 -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(0);
    // auto grad_input = indicate_middle * grad_output; 
    auto grad_alpha = grad_alpha_out.sum().unsqueeze(0);
    //TODO:to test use this way, later change into Gumble

    cudaDeviceSynchronize();
    clock_t time_LSQ_end = clock();

    // double quantize1_time = (double)(time_quantize1_end - time_quantize1_start) / CLOCKS_PER_SEC;
    double quantize_time = (double)(time_quantize_end - time_quantize_start) / CLOCKS_PER_SEC;
    double leverage_time = (double)(time_leverage_end - time_quantize_end) / CLOCKS_PER_SEC;
    double sample_time = (double)(time_sample_end - time_leverage_end) / CLOCKS_PER_SEC;
    // double sample1_time = (double)(time_method_start - time_leverage_end) / CLOCKS_PER_SEC;
    // double sample2_time = (double)(time_sampleflag_end - time_method_start) / CLOCKS_PER_SEC;
    // double sample3_time = (double)(time_sample_end - time_sampleflag_end) / CLOCKS_PER_SEC;
    double pack_time = (double)(time_pack_end - time_sample_end) / CLOCKS_PER_SEC;
    double gemm_time = (double)(time_gemm_end - time_pack_end) / CLOCKS_PER_SEC;
    double dequantize_time = (double)(time_dequantize_end - time_gemm_end) / CLOCKS_PER_SEC;
    double method1_time= 0, method2_time = 0, method3_time = 0;
    if (flag == 1){
        method1_time = (double)(time_dequantize_end - time_method_start) / CLOCKS_PER_SEC;
    }else if (flag == 2){
        if (cnt > 800){
            method2_time = (double)(time_dequantize_end - time_method_start) / CLOCKS_PER_SEC;
        } else{
            method3_time = (double)(time_dequantize_end - time_method_start) / CLOCKS_PER_SEC;
        }
    }
    // double pack_time = (double)(time_pack_end - time_sample_end) / CLOCKS_PER_SEC;
    // double int4gemm_time = (double)(time_int4gemm_end - time_pack_end) / CLOCKS_PER_SEC;
    // double fp16gemm_time = (double)(time_fp16gemm_end - time_int4gemm_end) / CLOCKS_PER_SEC;
    // double dequantize_time = (double)(time_dequantize_end - time_fp16gemm_end) / CLOCKS_PER_SEC;
    double LSQ_time = (double)(time_LSQ_end - time_dequantize_end) / CLOCKS_PER_SEC;
    // // time_leverage_end

    // time_vector.push_back(quantize1_time);
    time_vector.push_back(quantize_time);
    time_vector.push_back(leverage_time);
    time_vector.push_back(sample_time);
    // time_vector.push_back(sample1_time);
    // time_vector.push_back(sample2_time);
    // time_vector.push_back(sample3_time);
    time_vector.push_back(pack_time);
    time_vector.push_back(gemm_time);
    time_vector.push_back(dequantize_time);
    // time_vector.push_back(int4gemm_time);
    // time_vector.push_back(fp16gemm_time);
    // time_vector.push_back(dequantize_time);
    time_vector.push_back(LSQ_time);
    time_vector.push_back(method1_time);
    time_vector.push_back(method2_time);
    time_vector.push_back(method3_time);
    // auto sample_x = torch::cat({sample_x1, sample_x2}, 0);

    return std::make_tuple(grad_input, grad_alpha, grad_output, time_vector, first_transform, second_transform, x1_len, x2_len, scale1);
    // return std::make_tuple(gemm1, gemm2, gemm3, gemm4, sum_y1_column, sum_y2_column);
}
