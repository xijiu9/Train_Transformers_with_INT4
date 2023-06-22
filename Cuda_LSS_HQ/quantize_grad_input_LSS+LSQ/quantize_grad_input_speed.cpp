#include <torch/extension.h>

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_activation, torch::Tensor first_transform, torch::Tensor second_transform, torch::Tensor x1_len, torch::Tensor x2_len, float scale1);
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize_cuda(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_activation);


std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_activation, torch::Tensor first_transform, torch::Tensor second_transform, torch::Tensor x1_len, torch::Tensor x2_len, float scale1){
// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::vector<double>, int> quantize(torch::Tensor x, int num_bits, torch::Tensor qy, float scaley, torch::Tensor lsq_activation){

    return quantize_cuda(x, num_bits, qy, scaley, lsq_activation, first_transform, second_transform, x1_len, x2_len, scale1);
    // return quantize_cuda(x, num_bits, qy, scaley, lsq_activation);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("quantize", &quantize);
}