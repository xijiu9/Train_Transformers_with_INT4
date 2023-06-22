import torch
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import quantize_forward_easy

class MatrixConfig:
    def __init__(self):
        self.M = 4096
        self.K = 4096
        self.N = 4096
        self.testTurn = 30
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
cuda_tflops = []
cuda_hadamard_time = []
cuda_quantize_time = []
cuda_pack_time = []
cuda_gemm_time = []
cuda_dequantize_time = []
python_ordgemm_flops = []

class PreconditionerTest:
    def __init__(self):
        self.x = torch.randn(mconfig.M, mconfig.K).cuda().half()
        self.y = torch.randn(mconfig.N, mconfig.K).cuda().half()
        self.num_bins = 2 ** mconfig.num_bits - 1
        # self.hadamard = T[mconfig.group_size].repeat(mconfig.K // mconfig.group_size, 1)
        self.scale_hx = 0
        self.scale_hy = 0
        
    # x corresponds to input, y corresponds to weight
    # step1: hadamard quantize input and weight
    # step2: LSQ forward quantize input and weight, with scale_input and scale_weight
    
    
    def HadmardQuantize_cuda_speed(self, x, y):
        hadamard_time = 0
        quantize_time = 0
        pack_time = 0
        gemm_time = 0
        dequantize_time = 0
        
        hadamard = T[mconfig.group_size].half()
        x_shape = x.shape
        x_batch = x.view(-1,mconfig.group_size)
        y_shape = y.shape
        y_batch = y.view(-1,mconfig.group_size)
        
        total_time = 0
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            h_x = x_batch.matmul(hadamard).view(x_shape)
            h_y = y_batch.matmul(hadamard).view(y_shape)
            torch.cuda.synchronize()
            time_flag = time.time()
            # self.scale_hx = torch.tensor(self.scale_hx)
            out2 = quantize_forward_easy.quantize(h_x,h_y,self.scale_hx, self.scale_hy)
            torch.cuda.synchronize()
            time2 = time.time()
            if i>= 1:
                hadamard_time += time_flag - time1
                quantize_time += out2[5][0]
                pack_time += out2[5][1]
                gemm_time += out2[5][2]
                dequantize_time += out2[5][3]
                total_time += time2 - time1
        print("HQ cuda MM speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        # print("    output is:")
        # print(out2[0])
        print()
        # cuda_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        cuda_hadamard_time.append(hadamard_time)
        cuda_quantize_time.append(quantize_time)
        cuda_pack_time.append(pack_time)
        cuda_gemm_time.append(gemm_time)
        cuda_dequantize_time.append(dequantize_time)        
    
def draw_picture_full():
    plt.figure(figsize=(20, 20))
    area = plt.subplot2grid((11,11),(0,0),rowspan=11, colspan = 10)
    area.plot()
    
    bar_width = 0.6
    
    data = [cuda_hadamard_time, np.array(cuda_quantize_time), cuda_pack_time, cuda_gemm_time, cuda_dequantize_time]
    labels = ["hadamard", "quantize", "pack", "gemm", "dequantize"]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        
    r1 = range(len(matrix_shape))
    
    bottom_y = np.zeros(len(matrix_shape))
    data = np.array(data)
    sums = np.sum(data, axis=0)
    
    for index in range(len(data)):
        y = data[index] / sums
        plt.bar(r1, y, width=bar_width, edgecolor='white', label=labels[index], bottom=bottom_y, color=colors[index])
        bottom_y += y
        
    for i in range(data.shape[1]):
        tmp_y = 0
        for j in range(data.shape[0]):
            y = data[j][i] / sums[i]
            text = "%d%%" % (100 * y)
            plt.text(r1[i], tmp_y+y/2,text,color='white',size='40',horizontalalignment='center',verticalalignment='center')
            tmp_y += y
    
    plt.xticks(r1, matrix_shape, rotation=30, fontsize=45)
    plt.yticks(fontsize=60)

    # TODO: how to change legend place?
    # plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=50)
    # plt.legend(loc='upper center', bbox_to_anchor=(1,1), ncol=5, fontsize=45)

    plt.title("HQ", fontdict={'size' : 60})
    plt.ylabel('Time ratio', fontdict={'size' : 60})
    plt.xlabel("Matrix size (M,N,K)", fontdict={'size' : 60})

    plt.savefig('./image/HQ.svg', bbox_inches='tight', format='svg')
    # plt.savefig('./image/HQ.pdf', bbox_inches='tight')
    
if __name__=="__main__":
    
    # for (m,n,k) in [(9216, 10240, 12288),(10240, 12288,16384),(12288,12288,18432),(14336,13312,17408),(16384,15360,19456)]:
    for (m,n,k) in [(4608,5120,6144),(5120,6144,8192),(6144,6144,9216),(7168,6656,8704),(8192,7680,9728),(15360,8704,10752)]:
            print("matrix multiplication of {M,N,K} = {%d, %d, %d}" % (m,n,k))
            mconfig.M = m
            mconfig.N = n
            mconfig.K = k
            matrix_shape.append((mconfig.M, mconfig.N, mconfig.K))
            
            test = PreconditionerTest()
            test.HadmardQuantize_cuda_speed(test.x, test.y)
    
    draw_picture_full()