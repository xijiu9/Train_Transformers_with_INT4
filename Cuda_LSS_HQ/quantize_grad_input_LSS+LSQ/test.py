import torch
import qmatmul
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import quantize_grad_input_speed
import IPython

class MatrixConfig:
    def __init__(self):
        self.M = 4096
        self.K = 4096
        self.N = 4096
        self.testTurn = 1
        self.num_bits = 4
        self.group_size = 32
        
mconfig = MatrixConfig()

T = {}

size = 1
H = torch.ones(1, 1).cuda()
T[1] = H

for i in range(10):
    H = torch.cat((torch.cat([H, H], 1),
                   torch.cat([H, -H], 1)), 0) / math.sqrt(2)
    size *= 2
    T[size] = H
    
    
matrix_shape = []
cuda_speed_tflops = []
cuda_hadmard_time = []
cuda_quantize1_time = []
cuda_quantize2_time = []
cuda_leverage_time = []
cuda_sample_time = []
cuda_pack_time = []
cuda_gemm1_time = []
cuda_gemm2_time = []
cuda_dequantize_time = []

x1_list = []
y_list = []
class PreconditionerTest:
    def __init__(self):
        self.x = torch.randn(mconfig.M, mconfig.K).cuda().half() / 100
        self.y = torch.randn(mconfig.K, mconfig.N).cuda().half() / 100
        self.num_bins = 2 ** mconfig.num_bits - 1
        self.scale_y = max(abs(self.y.min()), abs(self.y.max())) / 7
        self.quantize_y = self.y / self.scale_y
        self.quantize_y.clamp_(-8.0, self.num_bins-8).round_()
        self.quantize_y = self.quantize_y.to(torch.int8)
        self.dequantize_y = self.quantize_y * self.scale_y
        self.zero_point1 = 0
        self.scale1 = 0
        self.zero_point2 = 0
        self.scale2 = 0
        self.hadmard = T[mconfig.group_size].half()
        self.activation = torch.randn(mconfig.M, mconfig.N).cuda().half() / 50
        self.hadamard_activation = self.activation.view(-1, mconfig.group_size).matmul(self.hadmard).view(self.activation.shape)
        self.scale_activation = torch.randn(1).cuda().half()
        
    def TwoLayerQuantizeInput_python(self, input):
        # x is grad_output, y is weight
        # backward quantize x, easy minimax quantize y
        total_time = 0
        for i in range(mconfig.testTurn + 1):
            time1 = time.time()
            mn = min(input.min() - 1e-8, 0).float()
            mx = max(input.max() + 1e-8, 0).float()
            
            self.zero_point1 = mn
            self.scale1 = self.num_bins / (mx - mn)
            
            qzero = -self.zero_point1 * self.scale1
            iqzero = torch.floor(qzero)
            
            if iqzero > 0:
                mx = (iqzero - self.num_bins) * mn / iqzero
            elif iqzero == 0:
                self.zero_point1, mn = 0, 0

            self.scale1 = self.num_bins / (mx - mn)
            
            first_transform = (input.float() - self.zero_point1) * self.scale1 - 8
            first_transform.clamp_(-8.0, self.num_bins-8).round_()
            first_quantize = ((first_transform+8) / self.scale1 + self.zero_point1).half()
            
            residual = input - first_quantize
            
            mn = min(residual.min() - 1e-8, 0).float()
            mx = max(residual.max() + 1e-8, 0).float()
            
            self.zero_point2 = mn
            self.scale2 = self.num_bins / (mx - mn)
            
            qzero = -self.zero_point2 * self.scale2
            iqzero = torch.floor(qzero)
                    
            if iqzero > 0:
                mx = (iqzero - self.num_bins) * mn / iqzero
            elif iqzero == 0:
                self.zero_point2, mn = 0, 0
            self.scale2 = self.num_bins / (mx - mn)
            second_transform = (residual.float() - self.zero_point2) * self.scale2 - 8
            noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
            second_transform.add_(noise)
            second_transform.clamp_(-8.0, self.num_bins-8).round_()
            second_quantize = ((second_transform+8) / self.scale2 + self.zero_point2).half()
            output = torch.cat([first_transform, second_transform], dim=0)
            output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
            
            # leverage score
            I = torch.eye(output_dequantize.shape[0] // 2).cuda()
            I2 = torch.cat([I,I], 0) 
            x_len = torch.linalg.norm(output_dequantize, dim=1)
            I_len = torch.linalg.norm(I2, dim=1)
            vec_norm = x_len.mul(I_len).float()
            len_norm = len(vec_norm)
            norm_activation = vec_norm / vec_norm.sum()
            small_num = norm_activation[:len_norm // 2].sum() * len_norm / 2 
            small_num = (small_num / 32).round() * 32
            if small_num > len_norm // 2:
                small_num = small_num - 32
            large_num = len_norm // 2 - small_num
            small_num = small_num.int()
            large_num = large_num.int()
            
            norm_activation = torch.log(norm_activation)
            # #Todo:currently Gumbel is not avaliable in libtorch
            activation_phi = torch.distributions.Gumbel(norm_activation, torch.ones_like(norm_activation)).rsample()
            # activation_phi = norm_activation
            # IPython.embed()
            # Todo:test the correctness of cuda
            self.norm_activation = norm_activation
            self.activation_phi = activation_phi
            
            small_values, small_indices = torch.topk(activation_phi[:len(norm_activation) // 2], small_num)   
            large_values, large_indices = torch.topk(activation_phi[len(norm_activation) // 2:], large_num)
            
            index = torch.cat([small_indices, large_indices + len(norm_activation) // 2])
            
            cnt = 0
            norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum())

            while norm_activation_loop.max() > 1 and cnt < len_norm / 2:
                small_index = torch.nonzero((norm_activation_loop < 1)).squeeze()
                small_value = norm_activation_loop[small_index]
                cnt = len_norm - len(small_index)
                norm_activation_loop = torch.clamp(norm_activation_loop, 0, 1)
                if small_value.max() == 0 and small_value.min() == 0:
                    break
                small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
                norm_activation_loop[small_index] = small_value
            
            # sample_index = torch.bernoulli(norm_activation_loop)
            # index = torch.nonzero((sample_index == 1)).squeeze()
            norm_activation_loop[norm_activation_loop == 0] = 1e-10
            # output = output / norm_activation_loop.unsqueeze(1)
            output_dequantize = output_dequantize / norm_activation_loop.unsqueeze(1)
            # Ind = torch.zeros_like(output)
            Ind = torch.zeros_like(output_dequantize)
            Ind[index] = 1
            # output = output.mul(Ind)
            output_dequantize = output_dequantize.mul(Ind)
            
            
            # dequantize inputx
            # first = (output[0:output.shape[0] // 2]+8) / self.scale1 + self.zero_point1
            # second = (output[output.shape[0] // 2:]+8) / self.scale2 + self.zero_point2
            # dequantize_sample_x = (first + second).half()
            dequantize_sample_x = (output_dequantize[0:output_dequantize.shape[0] // 2] + output_dequantize[output_dequantize.shape[0] // 2:]).half()
            
            # dequantize inputy
            dequantize_sample_y = self.quantize_y * self.scale_y 
            grad_output = dequantize_sample_x.matmul(dequantize_sample_y)
            
            # calculate grad_activation and grad_scale_activation through LSQ
            q_w = self.hadamard_activation / self.scale_activation
            indicate_small = (q_w < -8).half()
            indicate_big = (q_w > 7).half()
            indicate_middle = 1.0 - indicate_small - indicate_big
            grad_scale = 1.0 / math.sqrt(self.hadamard_activation.numel() * 7)
            grad_alpha = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
                    -q_w + q_w.round())) * grad_output * grad_scale).sum().unsqueeze(dim=0)
            #Todo:need to matmul a hadamard matrix?
            grad_input = indicate_middle * grad_output
            # import IPython
            # IPython.embed()
            
            # test
            # first_test = first_quantize.matmul(dequantize_sample_y)
            # second_test = second_quantize.matmul(dequantize_sample_y)
            # Ind2_high = torch.zeros_like(h_output)
            # Ind2_low = torch.zeros_like(h_output)
            # Ind2_high[0:quart_size] = 1
            # Ind2_low[quart_size:] = 1
            # test_out = first_test.mul(Ind2_high) + second_test.mul(Ind2_low)
            
            sample_x1 = output[0:output.shape[0] // 2].half()
            sample_x2 = output[output.shape[0] // 2:].half()
            Ind_small = Ind[:Ind.shape[0] // 2]
            Ind_large = Ind[Ind.shape[0] // 2:]
            Ind_small_index = Ind_small[small_indices,...]
            sample_y = dequantize_sample_y
            # IPython.embed()
            sample_y_tmp = self.quantize_y
            sample_x1_tmp = first_transform.mul(Ind_small.half())
            gemm1_tmp = sample_x1_tmp.half().matmul(self.quantize_y.half())
            sample_x2_tmp = second_transform.mul(Ind_large.half())
            gemm2_tmp = sample_x2_tmp.half().matmul(self.quantize_y.half())
            gemm1 = sample_x1.matmul(sample_y)
            gemm2 = sample_x2.matmul(sample_y)
            torch.cuda.synchronize()
            time2 = time.time()
            if i >= 1:
                total_time += time2 - time1
                
            
        x1_list.append(sample_x1_tmp)
        y_list.append(sample_y_tmp)
        print("quantize python:")
        torch.set_printoptions(precision=4)
        # print("sample_x1 is:")
        # print(sample_x1_tmp)
        # print("sample_x2 is:")
        # print(sample_x2_tmp)
        # print("sample_y is:")
        # print(sample_y_tmp)
        # print("small index is:")
        # print(small_indices)
        # print("Ind_small:")
        # print(Ind_small)
        # print("Ind_small_index:")
        # print(Ind_small_index)
        # print("gemm1 is:")
        # print(gemm1_tmp)
        # print("gemm1 is:")
        # print(gemm1)
        # print("gemm2 is:")
        # print(gemm2)
        # print("activation_phi is:")
        # print(activation_phi)
        # print("norm activation is:")
        # print(norm_activation)
        print("grad_output is:")
        print(grad_output)
        print("grad of scale_activation is:")
        print(grad_alpha)
        # print("grad of activation is:")
        # print(h_grad_input)
        # print("grad_scale is:")
        # print(grad_scale)
        # print("indicate_middle is:")
        # print(indicate_middle)
        # import IPython
        # IPython.embed()
        # print("h_output is:")
        # print(h_output)
        # print("test h_out is:")
        # print(test_out)
        
    
    def TwoLayerQuantizeInput_cuda_speed(self, input):
        total_time = 0
        hadmard_time = 0
        quantize1_time = 0
        quantize2_time = 0
        leverage_time = 0
        sample_time = 0
        pack_time = 0
        gemm1_time = 0
        gemm2_time = 0
        dequantize_time = 0
        
        y_shape = self.y.shape
        y_batch = self.y.view(-1,mconfig.group_size)
        
        for i in range(mconfig.testTurn + 1):
            time1 = time.time()
            
            # hy = y_batch.matmul(self.hadmard).view(y_shape)
            qmatmul.synchronize()
            time_flag = time.time()
            
            # mn = min(input.min() - 1e-8, 0)
            # mx = max(input.max() + 1e-8, 0)
            
            # self.zero_point1 = mn
            # # self.scale1 = self.num_bins / (mx - mn)
            
            # qzero = -self.zero_point1 * self.scale1
            # iqzero = torch.floor(qzero)
            
            # if iqzero > 0:
            #     mx = (iqzero - self.num_bins) * mn / iqzero
            # elif iqzero == 0:
            #     self.zero_point1, mn = 0, 0
                
            # self.scale1 = self.num_bins / (mx - mn)
            # first_quantize = quantize_grad_input_speed.first_quantize(input, self.scale1, self.zero_point1)
            qmatmul.synchronize()
            time2 = time.time()
            
            # residual = input - first_quantize[1]
            
            # mn = min(residual.min() - 1e-8, 0)
            # mx = max(residual.max() + 1e-8, 0)
            
            # self.zero_point2 = mn
            # # self.scale2 = self.num_bins / (mx - mn)
            
            # qzero = -self.zero_point2 * self.scale2
            # iqzero = torch.floor(qzero)
                    
            # if iqzero > 0:
            #     mx = (iqzero - self.num_bins) * mn / iqzero
            # elif iqzero == 0:
            #     self.zero_point2, mn = 0, 0
            # self.scale2 = self.num_bins / (mx - mn)
            
            qmatmul.synchronize()
            time3 = time.time()
            first_out = quantize_grad_input_speed.first_quantize(input, self.dequantize_y, self.num_bins)
            # IPython.embed()
            activation_phi = torch.distributions.Gumbel(self.norm_activation, torch.ones_like(self.norm_activation)).rsample()
            activation_phi = self.activation_phi
            second_transform = quantize_grad_input_speed.second_quantize(first_out[1], first_out[2],first_out[3],first_out[4],first_out[5],first_out[6],
                                                                         first_out[7],first_out[8],first_out[9],first_out[10],first_out[11],first_out[12], first_out[13],                                                                 
                                                                         first_out[14], first_out[15],activation_phi, self.quantize_y, self.scale_y, 
                                                                         self.hadamard_activation, self.scale_activation)
            qmatmul.synchronize()
            time_flag2 = time.time()
            qmatmul.synchronize()
            time4 = time.time()
            # if i >= 1:
            #     total_time += time4 - time1
            #     hadmard_time += (time_flag - time1) + (time4 - time_flag2)
            #     quantize1_time += time2 - time1
            #     quantize2_time += second_transform[1][0] + (time3 - time2)
            #     leverage_time += second_transform[1][1]
            #     sample_time += second_transform[1][2]
            #     pack_time += second_transform[1][3]
            #     gemm1_time += second_transform[1][4]
            #     gemm2_time += second_transform[1][5]
                # dequantize_time += second_transform[1][6]
                
        # x1_list.append(second_transform[2])
        # y_list.append(second_transform[3].t())
        print("quantize cuda speed:")
        torch.set_printoptions(precision=4)
        # print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        # print("gemm1 is:")
        # print(second_transform[0])
        # print("gemm2 is:")
        # print(second_transform[1])
        # print("norm activation is:")
        # print(first_out[0])
        # print("activation_phi is:")
        # print(activation_phi)
        print("grad_output is:")
        print(second_transform[2])
        # print("indicate_middle is:")
        # print(second_transform[3])
        print("grad of scale_activation is:")
        print(second_transform[1])
        # print("grad of activation is:")
        # print(second_transform[2])
        # print("grad_scale is:")
        # print(second_transform[5])
        # print("h_output is:")
        # print(second_transform[0])
        # print("sample_x1 is:")
        # print(second_transform[2])
        # print("sample_x2 is:")
        # print(second_transform[3])
        # print("sample_y is:")
        # print(second_transform[3].t())
        # print("small index is:")
        # print(second_transform[2])
        # print("Ind_small:")
        # print(second_transform[4])
        # print("Ind_small_index:")
        # print(second_transform[4])
        
        print()
        # cuda_speed_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        cuda_hadmard_time.append(hadmard_time)
        cuda_quantize1_time.append(quantize1_time)
        cuda_quantize2_time.append(quantize2_time)
        cuda_leverage_time.append(leverage_time)
        cuda_sample_time.append(sample_time)
        cuda_pack_time.append(pack_time)
        cuda_gemm1_time.append(gemm1_time)
        cuda_gemm2_time.append(gemm2_time)
        cuda_dequantize_time.append(dequantize_time)
        IPython.embed()
        

if __name__=="__main__":
    # for (m,n,k) in [(512,512,1024),(2048,1024,1024),(4096,2048,2048),(4096,2560,3584),(5120,6144,8192),(6144,6144,9216),(15360,8704,10752)]:
    # for (m,n,k) in [(512,512,1024),(2048,1024,1024),(4096,2048,2048),(4096,2560,3584)]:
    for (m,n,k) in [(4608, 5120, 6144)]:
    # for (m,n,k) in [(512,512,1024)]:
        print("matrix multiplication of {M,N,K} = {%d, %d, %d}" % (m,n,k))
        mconfig.M = m
        mconfig.N = n
        mconfig.K = k
        matrix_shape.append((mconfig.M, mconfig.N, mconfig.K))
        test = PreconditionerTest()
        test.TwoLayerQuantizeInput_python(test.x)
        test.TwoLayerQuantizeInput_cuda_speed(test.x)