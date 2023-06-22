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
#include "torch/script.h"
#include "cuda_runtime.h"
#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include <curand_kernel.h>

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

//Todo:output high corresponds to scale1 and zero1, low corresponds to scale2 and zero2
template<typename scalar_t>
__global__ void dequantize_cuda_kernel(const int32_t * gemm1, const int32_t * gemm2, const float * norm_small, const float * norm_large,
                                        scalar_t * __restrict__ grad_output, const float scale_gemm1, const float scale_gemm2, long long int size, int ny){
    // extern __shared__ float s[];
    // float * y_col = s;  // N_THREADS float
    
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    int row = x / ny;
    // , col = x - row * ny
    float norm1 = norm_small[row];
    float norm2 = norm_large[row];
    // bool index1 = small_indices[row];
    // bool index2 = large_indices[row];
    // int64_t sumY = sum_y_column[col];

    // y_col[threadIdx.x] = sum_y_column[col];
    // __syncthreads();

    if (x<size){
        float smallValue = (gemm1[x] * scale_gemm1) / norm1;
        float largeValue = (gemm2[x] * scale_gemm2) / norm2;
        grad_output[x] = smallValue + largeValue;
    }
}

template<typename scalar_t>
__global__ void norm_cuda_kernel(const float * norm_small, const float * norm_large, const scalar_t * __restrict__ output_low, const scalar_t * __restrict__ output_high, 
                                scalar_t * __restrict__ grad_output, long long int size, int ny){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;
    int row = x / ny;
    float norm1 = norm_small[row];
    float norm2 = norm_large[row];

    if (x<size){
    //    output[x] = (gemm1[x] * scale_gemm1 + const_x1 * sumY) / norm1 + (gemm2[x] * scale_gemm2 + const_x2 * sumY) / norm2;
        grad_output[x] = output_low[x] / norm1 + output_high[x] / norm2;
    }
}

template<typename scalar_t>
__global__ void multiple_kernel(const scalar_t * __restrict__ in, scalar_t * __restrict__ out, float scale, long long int size){
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x<size){
        out[x] = in[x] * scale;
    }
}

template<typename scalar_t>
__global__ void LSQ_cuda_kernel(const scalar_t * lsq_activation, const scalar_t * __restrict__ grad_output, scalar_t * __restrict__ grad_alpha_out, 
                                scalar_t * __restrict__ grad_input, const float grad_scale, const long long int size){  
    long long int x = threadIdx.x + blockIdx.x * blockDim.x;

    if (x<size){
       scalar_t q_w = lsq_activation[x];
       scalar_t indicate_small = (q_w < -8);
       scalar_t indicate_big = (q_w > 7);
       scalar_t indicate_middle = 1.0 - indicate_small - indicate_big;
       scalar_t grad_out = grad_output[x];
       grad_alpha_out[x] = (indicate_small * -8 + indicate_big * 7 + indicate_middle * (-q_w + round(q_w))) * grad_out * grad_scale;
    //    grad_alpha_out[x] = 0;
       grad_input[x] = indicate_middle * grad_out;
    }
}

__device__ __inline__ c10::Half __shfl_down_sync(const unsigned mask, const c10::Half var,
                                                 const unsigned int delta, const int width) {
  __half var_ = var;
  return __shfl_down_sync(mask, var_, delta, width);
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

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_activation, torch::Tensor first_transform, torch::Tensor second_transform, torch::Tensor x1_len, torch::Tensor x2_len, float scale1){
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_activation){
    std::vector<double> time_vector;
    long long int nx = x.size(0);
    long long int nz = x.size(1);
    long long int ny = qy.size(1);

    cudaDeviceSynchronize();
    clock_t time_quantize_start = clock();

    auto option_transform = torch::TensorOptions().dtype(torch::kInt8).device(x.device());
    auto option_quantize = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    auto option_float = torch::TensorOptions().dtype(torch::kFloat32).device(x.device());
    auto option_int = torch::TensorOptions().dtype(torch::kInt32).device(x.device());
    auto option_output = torch::TensorOptions().dtype(x.dtype()).device(x.device());
    // torch::Tensor first_transform = torch::empty({nx, nz}, option_transform);
    // torch::Tensor first_quantize = torch::empty({nx, nz}, option_quantize);
    // torch::Tensor second_transform = torch::empty({nx, nz}, option_transform);
    // torch::Tensor second_quantize = torch::empty({nx, nz}, option_quantize);
    
    dim3 block(N_THREADS);
    dim3 grid1((nx*nz-1)/block.x+1);
    // int size_quantize = nx * nz ;
    // // process of first quantize
    // float mn = std::min(x.min().item<float>() - 1e-8, 0.);
    // float mx = std::max(x.max().item<float>() + 1e-8, 0.);

    int num_bins_half = pow(2, num_bits) - 2;
    // int num_bins = num_bins_half * num_bins_half;
    // int num_bins_clamp = num_bins_half / 2 - 1;

    // float scale1 = num_bins / (2 * max(fabs(mn), fabs(mx)));

    // cudaDeviceSynchronize();
    // clock_t time_quantize1_end = clock();

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(x.scalar_type(), "quantize_cuda", ([&] {
    // quantize_cuda_kernel<scalar_t><<<grid1, block>>>(
    //     x.data_ptr<scalar_t>(),
    //     first_transform.data_ptr<int8_t>(),
    //     // first_quantize.data_ptr<scalar_t>(),
    //     second_transform.data_ptr<int8_t>(),
    //     // second_quantize.data_ptr<scalar_t>(),
    //     num_bins_half, num_bins_clamp,
    //     scale1, size_quantize,rand());
    // }));

    cudaDeviceSynchronize();
    clock_t time_quantize_end = clock();

    // leverage score
    // TODO: use dim=0 because torch.linalg only supports dim=1
    // int threads = 32;
    // int blocks = nx;
    // // auto x_sample = torch::cat({first_quantize, second_quantize}, 0);
    // auto x1_len = torch::empty({nx,}, option_float);
    // auto x2_len = torch::empty({nx,}, option_float);

    // int stride_x = nz / 32;
    // float scale_x1 = num_bins_half / scale1;
    // float scale_x2 = 1. / scale1;

    // linalg_normInt_cuda_kernel<<<blocks, threads>>>(
    //     first_transform.data_ptr<int8_t>(), 
    //     x1_len.data_ptr<float>(),
    //     nx,nz,stride_x, scale_x1);

    // linalg_normInt_cuda_kernel<<<blocks, threads>>>(
    //     second_transform.data_ptr<int8_t>(), 
    //     x2_len.data_ptr<float>(),
    //     nx,nz,stride_x, scale_x2);

    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(first_quantize.scalar_type(), "linalg_cuda", ([&] {
    // linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
    //     first_quantize.data_ptr<scalar_t>(), 
    //     x1_len.data_ptr<float>(),
    //     nx,nz,stride_x);
    // }));
    // AT_DISPATCH_FLOATING_TYPES_AND_HALF(second_quantize.scalar_type(), "linalg_cuda", ([&] {
    // linalg_norm_cuda_kernel<scalar_t><<<blocks, threads>>>(
    //     second_quantize.data_ptr<scalar_t>(), 
    //     x2_len.data_ptr<float>(),
    //     nx,nz,stride_x);
    // }));
    auto vec_norm = torch::cat({x1_len, x2_len});
    long long int len_norm = vec_norm.numel();

    cudaDeviceSynchronize();
    clock_t time_leverage_end = clock();
    
    int cnt = 0;
    int whilenum = 0;
    // auto norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum());
    auto norm_activation_loop = torch::empty_like(vec_norm);
    float scale_norm = len_norm / (2 * vec_norm.sum().item<float>());
    dim3 grid_norm(len_norm/block.x+1);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(vec_norm.scalar_type(), "multiple_cuda", ([&] {
    multiple_kernel<scalar_t><<<grid_norm, block>>>(
        vec_norm.data_ptr<scalar_t>(),
        norm_activation_loop.data_ptr<scalar_t>(),
        scale_norm,len_norm);
    }));
    auto sample_index = torch::empty_like(norm_activation_loop);
    // int posNum = (norm_weight_loop > 0).sum().item<int>();
    // cudaDeviceSynchronize();
    // clock_t time_sample1_end = clock();

    // TODO:change back
    // if ((norm_activation_loop > 0).sum().item<int>() < len_norm / 2){
    if (true) {
    // if (false) {
        norm_activation_loop.index_put_({norm_activation_loop > 0}, 1);
        sample_index = norm_activation_loop;
    } else {
        bool whileloop = (norm_activation_loop.max() > 1).item<bool>();
        while (whileloop && cnt < len_norm / 2){
            auto small_index = (norm_activation_loop < 1);
            auto small_value = norm_activation_loop.index({small_index});
            long long int small_len = small_value.numel();
            cnt = len_norm - small_len;
            norm_activation_loop = torch::clamp(norm_activation_loop, 0, 1);
            bool breakloop = (small_value.max() == 0).item<bool>();
            if (breakloop)
                break;
            // small_value = small_value * (len_norm / 2 - cnt) / small_value.sum();
            float scale_small = (len_norm / 2 - cnt) / small_value.sum().item<float>();
            dim3 grid_small(small_len/block.x+1);
            AT_DISPATCH_FLOATING_TYPES_AND_HALF(small_value.scalar_type(), "multiple_cuda", ([&] {
            multiple_kernel<scalar_t><<<grid_small, block>>>(
                small_value.data_ptr<scalar_t>(),
                small_value.data_ptr<scalar_t>(),
                scale_small,small_len);
            }));
            // norm_activation_loop[small_index] = small_value;
            norm_activation_loop.index_put_({small_index}, small_value);
            whileloop = (norm_activation_loop.max() > 1).item<bool>();
            whilenum++;
        } 
        sample_index = torch::bernoulli(norm_activation_loop);
    }
    auto small_indices = torch::nonzero(sample_index.index({Slice({None, len_norm/2})}) == 1).squeeze(1);
    auto large_indices = torch::nonzero(sample_index.index({Slice(len_norm/2)}) == 1).squeeze(1);

    norm_activation_loop.index_put_({norm_activation_loop == 0}, 1);
    // auto left_indices = (sample_index != 1);

    // norm_activation_loop.index_put_({norm_activation_loop == 0}, 1);
    // sample process
    dim3 grid2((nx*ny-1)/block.x+1);
    long long int size = nx*ny;

    int small_num_ = small_indices.numel();
    int large_num_ = large_indices.numel();

    auto sample_x1 = first_transform.index({small_indices});
    auto sample_x2 = second_transform.index({large_indices});
    auto sample_y = qy.t().contiguous();
    
    cudaDeviceSynchronize();
    clock_t time_sample_end = clock();

    // pack process
    auto sample_x1_int4 = torch::empty({small_num_, nz>>1}, option_transform);
    auto sample_x2_int4 = torch::empty({large_num_, nz>>1}, option_transform);
    auto sample_y_int4 = torch::empty({ny, nz>>1}, option_transform);
    long long int grid_size_x1 = nz*small_num_/2;
    long long int grid_size_x2 = nz*large_num_/2;
    long long int grid_size_y = nz*ny/2;
    dim3 grid_pack_x1((grid_size_x1-1)/block.x+1);
    dim3 grid_pack_x2((grid_size_x2-1)/block.x+1);
    dim3 grid_pack_y((grid_size_y-1)/block.x+1);
    if (small_num_ > 0) {
        pack_cuda_kernel<<<grid_pack_x1,block>>>(sample_x1.data_ptr<int8_t>(), sample_x1_int4.data_ptr<int8_t>(), grid_size_x1);
    }
    if (large_num_ > 0) {
        pack_cuda_kernel<<<grid_pack_x2,block>>>(sample_x2.data_ptr<int8_t>(), sample_x2_int4.data_ptr<int8_t>(), grid_size_x2);
    }
    pack_cuda_kernel<<<grid_pack_y,block>>>(sample_y.data_ptr<int8_t>(), sample_y_int4.data_ptr<int8_t>(), grid_size_y);

    cudaDeviceSynchronize();
    clock_t time_pack_end = clock();


    cudaError_t result;
    int lda = nz;
    int ldb = nz;
    int ldc = ny;
    // Chunked matrix multiplication
    auto gemm1 = torch::empty({small_num_,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    result = CutlassSgemmNN(small_num_, ny, nz, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm1.data_ptr<int32_t>(), ldc);
    
    auto gemm2 = torch::empty({large_num_,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    result = CutlassSgemmNN(large_num_, ny, nz, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm2.data_ptr<int32_t>(), ldc);

    cudaDeviceSynchronize();
    clock_t time_gemm_end = clock();

    auto gemm1_out = torch::zeros({nx,ny}, option_int);
    auto gemm2_out = torch::zeros({nx,ny}, option_int);
    gemm1_out.index_put_({small_indices}, gemm1);
    gemm2_out.index_put_({large_indices}, gemm2);

    // // gemm process
    // cudaError_t result;
    // int lda = nz;
    // int ldb = nz;
    // int ldc = ny;
    // // Chunked matrix multiplication
    // auto gemm1 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    // result = CutlassSgemmNN(nx, ny, nz, reinterpret_cast<cutlass::int4b_t *>(sample_x1_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm1.data_ptr<int32_t>(), ldc);

    // auto gemm2 = torch::empty({nx,ny}, at::device(at::kCUDA).dtype(torch::kInt32));
    // result = CutlassSgemmNN(nx, ny, nz, reinterpret_cast<cutlass::int4b_t *>(sample_x2_int4.data_ptr<int8_t>()), lda, reinterpret_cast<cutlass::int4b_t *>(sample_y_int4.data_ptr<int8_t>()), ldb, gemm2.data_ptr<int32_t>(), ldc);

    // dequantize process
    // auto sum_y_column = torch::sum(qy, 0);
    // auto output_low = torch::empty({nx,ny}, option_output);
    // auto output_high = torch::empty({nx,ny}, option_output);
    auto grad_output = torch::empty({nx,ny}, option_output);

    // float const_x1 = (8.0 / scale1 + zero_point1) * scaley;
    // float const_x2 = (8.0 / scale2 + zero_point2) * scaley;
    float scale_gemm1 = scaley * num_bins_half / (scale1);
    float scale_gemm2 = scaley / (scale1);
    auto norm_small = norm_activation_loop.index({Slice({None, len_norm/2})});
    auto norm_large = norm_activation_loop.index({Slice(len_norm/2)});
    // auto small_indices = left_indices.index({Slice({None, len_norm/2})});
    // auto large_indices = left_indices.index({Slice(len_norm/2)});


    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "dequantize_cuda", ([&] {
    // dequantize_cuda_kernel<scalar_t><<<grid2, block, N_THREADS * sizeof(float)>>>(
    dequantize_cuda_kernel<scalar_t><<<grid2, block>>>(
        gemm1_out.data_ptr<int32_t>(), 
        gemm2_out.data_ptr<int32_t>(),
        norm_small.data_ptr<float>(),
        norm_large.data_ptr<float>(),
        grad_output.data_ptr<scalar_t>(),
        // sum_y_column.data_ptr<int64_t>(),
        scale_gemm1, scale_gemm2,
        size, ny);
    }));

    cudaDeviceSynchronize();
    clock_t time_dequantize_end = clock();

    // auto grad_output = output_low;
    float grad_scale = 1.0 / sqrt(lsq_activation.numel() * 7);
    auto grad_alpha_out = torch::empty({nx,ny}, option_output);
    auto grad_input = torch::empty({nx,ny}, option_output);
    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_output.scalar_type(), "LSQ_cuda", ([&] {
    LSQ_cuda_kernel<scalar_t><<<grid2, block>>>(
        lsq_activation.data_ptr<scalar_t>(), 
        grad_output.data_ptr<scalar_t>(),
        grad_alpha_out.data_ptr<scalar_t>(),
        grad_input.data_ptr<scalar_t>(),
        grad_scale, size);
    }));

    auto grad_alpha = grad_alpha_out.sum().unsqueeze(0);
    cudaDeviceSynchronize();
    clock_t time_LSQ_end = clock();


    double quantize_time = (double)(time_quantize_end - time_quantize_start) / CLOCKS_PER_SEC;
    // double quantize2_time = (double)(time_quantize_end - time_quantize1_end) / CLOCKS_PER_SEC;
    double leverage_time = (double)(time_leverage_end - time_quantize_end) / CLOCKS_PER_SEC;
    double sample_time = (double)((time_sample_end - time_leverage_end)) / CLOCKS_PER_SEC;
    double pack_time = (double)(time_pack_end - time_sample_end) / CLOCKS_PER_SEC;
    double gemm_time = (double)(time_gemm_end - time_pack_end) / CLOCKS_PER_SEC;
    double dequantize_time = (double)(time_dequantize_end - time_gemm_end) / CLOCKS_PER_SEC;
    double LSQ_time = (double)(time_LSQ_end - time_dequantize_end) / CLOCKS_PER_SEC;
    // double LSS_time = (double)(time_LSS_end - time_dequantize_end) / CLOCKS_PER_SEC;
    // // time_leverage_end

    time_vector.push_back(quantize_time);
    // time_vector.push_back(quantize2_time);
    time_vector.push_back(leverage_time);
    time_vector.push_back(sample_time);
    time_vector.push_back(pack_time);
    time_vector.push_back(gemm_time);
    time_vector.push_back(dequantize_time);
    time_vector.push_back(LSQ_time);

    return std::make_tuple(grad_input, grad_alpha, gemm1, gemm2, time_vector, whilenum);
}
