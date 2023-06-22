import torch
import quantize_grad_weight_speed
import time
import matplotlib.pyplot as plt
import numpy as np
import math
import quantize_forward_easy
import quantize_grad_input_speed


class MatrixConfig:
    def __init__(self):
        self.M = 4096
        self.K = 4096
        self.N = 4096
        self.testTurn = 20
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
cuda_easy_tflops = []
twolayer_cuda_speed_tflops = []
twolayerInput_cuda_speed_tflops = []
hadamard_cuda_speed_tflops = []
cuda_hadamard_time = []
cuda_quantize_time = []
cuda_leverage_time = []
cuda_sample_time = []
cuda_pack_time = []
cuda_gemm_time = []
cuda_dequantize_time = []
cuda_LSQ_time = []
python_ordgemm_flops = []
cuda_input_quantize_time = []
cuda_input_leverage_time = []
cuda_input_sample_time = []
cuda_input_pack_time = []
cuda_input_gemm_time = []
cuda_input_dequantize_time = []
cuda_input_LSQ_time = []

phi_list = []
small_index_list = []
norm_list = []
first_transform_list = []
class PreconditionerTest:
    def __init__(self):
        self.x1 = torch.randn(int(mconfig.K/2), mconfig.M).cuda().half() / 100
        self.x2 = torch.zeros_like(self.x1)
        self.x = torch.cat([self.x1, self.x2], 0)
        self.y = torch.randn(mconfig.K, mconfig.N).cuda().half() / 100
        self.y_input = torch.randn(mconfig.M, mconfig.N).cuda().half() / 100
        self.num_bins = 2 ** mconfig.num_bits - 1
        self.num_bits = mconfig.num_bits
        
        self.scale_y = max(abs(self.y.min()), abs(self.y.max())) / 7
        self.quantize_y = self.y / self.scale_y
        self.quantize_y.clamp_(-8.0, self.num_bins-8).round_()
        self.quantize_y = self.quantize_y.to(torch.int8)
        self.dequantize_y = self.quantize_y * self.scale_y
        
        self.scale_yinput = max(abs(self.y_input.min()), abs(self.y_input.max())) / 7
        self.quantize_yinput = self.y_input / self.scale_yinput
        self.quantize_yinput.clamp_(-8.0, self.num_bins-8).round_()
        self.quantize_yinput = self.quantize_yinput.to(torch.int8)
        self.dequantize_yinput = self.quantize_yinput * self.scale_yinput
        
        self.hadamard = T[mconfig.group_size].half()
        self.weight = torch.randn(mconfig.M, mconfig.N).cuda().half() / 50
        self.hadamard_weight = self.weight.view(-1, mconfig.group_size).matmul(self.hadamard).view(self.weight.shape)
        self.scale_weight = torch.randn(1).cuda().half()
        self.lsq_weight = self.weight / self.scale_weight
        
        
        self.input = torch.randn(mconfig.K, mconfig.N).cuda().half() / 50
        self.hadamard_input = self.input.view(-1, mconfig.group_size).matmul(self.hadamard).view(self.input.shape)
        self.scale_input = torch.randn(1).cuda().half()
        self.lsq_input = self.input / self.scale_input
        
    def TwoLayerQuantizeInput_cuda_speed(self, input, inputList):
        total_time = 0
        quantize_time = 0
        leverage_time = 0
        sample_time = 0
        pack_time = 0
        gemm_time = 0
        dequantize_time = 0
        LSQ_time = 0
        
        
        for i in range(mconfig.testTurn + 1):
            torch.cuda.synchronize()
            time1 = time.time()
            
            activation_out = quantize_grad_input_speed.quantize(input, self.num_bits, self.quantize_yinput, self.scale_yinput, self.lsq_input, inputList[0], inputList[1], inputList[2], inputList[3],inputList[4])
            torch.cuda.synchronize()
            time2 = time.time()
            if i >= 1:
                quantize_time += activation_out[4][0]
                leverage_time += activation_out[4][1]
                sample_time += activation_out[4][2]
                pack_time += activation_out[4][3]
                gemm_time += activation_out[4][4]
                dequantize_time += activation_out[4][5]
                LSQ_time += activation_out[4][6]
                total_time += time2 - time1
                
        
        print("quantize cuda speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        print("total_time is:")
        print(total_time)
        print("gemm_time is:")
        print(gemm_time)
        print("dequantize_time is:")
        print(dequantize_time)
        
        twolayerInput_cuda_speed_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        cuda_input_quantize_time.append(quantize_time)
        cuda_input_leverage_time.append(leverage_time)
        cuda_input_sample_time.append(sample_time)
        cuda_input_pack_time.append(pack_time)
        cuda_input_gemm_time.append(gemm_time)
        cuda_input_dequantize_time.append(dequantize_time)
        cuda_input_LSQ_time.append(LSQ_time)
        
    def TwoLayerQuantizeWeight_cuda_speed(self, input):
        total_time = 0
        method2_time = 0
        method3_time = 0
        quantize_time = 0
        leverage_time = 0
        sample_time = 0
        pack_time = 0
        gemm_time = 0
        dequantize_time = 0
        LSQ_time = 0
        
        for i in range(mconfig.testTurn + 1):
            torch.cuda.synchronize()
            time1 = time.time()

            weight_out = quantize_grad_weight_speed.quantize(input, self.num_bits, self.quantize_y, self.scale_y, self.lsq_weight)
            # assert torch.isnan(weight_out[0]).sum() == 0
            torch.cuda.synchronize()
            time2 = time.time()
            if i >= 1:
                total_time += time2 - time1
                quantize_time += weight_out[3][0]
                leverage_time += weight_out[3][1]
                sample_time += weight_out[3][2]
                pack_time += weight_out[3][3]
                gemm_time += weight_out[3][4]
                dequantize_time += weight_out[3][5]
                LSQ_time += weight_out[3][6]
                method2_time += weight_out[3][8]
                method3_time += weight_out[3][9]
        
                
        # first_transform_list.append(second_transform[1])
        print("LSS cuda MM speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        print("total_time is:")
        print(total_time)
        print("gemm_time is:")
        print(gemm_time)
        # print("sample1_time is:")
        # print(sample1_time)
        # print("sample2_time is:")
        # print(sample2_time)
        # print("sample3_time is:")
        # print(sample3_time)
        # print("small num is:")
        # print(weight_out[-2])
        # print("large num is:")
        # print(weight_out[-1])
        twolayer_cuda_speed_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        cuda_quantize_time.append(quantize_time)
        cuda_leverage_time.append(leverage_time)
        cuda_sample_time.append(sample_time)
        cuda_pack_time.append(pack_time)
        cuda_gemm_time.append(gemm_time)
        cuda_dequantize_time.append(dequantize_time)
        cuda_LSQ_time.append(LSQ_time)
        # import IPython
        # IPython.embed()
        return weight_out[-5], weight_out[-4], weight_out[-3], weight_out[-2], weight_out[-1]
        
    def Gemm_ordinary_python(self, x, y):
        total_time = 0
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            out = x.t().matmul(y)
            torch.cuda.synchronize()
            time2 = time.time()
            if i>= 1:
                total_time += time2 - time1
        print("fp16 gemm speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        print()
        python_ordgemm_flops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        
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
        h_x = x_batch.matmul(hadamard).view(x_shape)
        h_y = y_batch.matmul(hadamard).view(y_shape)
        scale_hx = max(abs(h_x.max()), abs(h_x.min())) / 7
        scale_hy = max(abs(h_y.max()), abs(h_y.min())) / 7
        for i in range(mconfig.testTurn+1):
            time1 = time.time()
            h_x = x_batch.matmul(hadamard).view(x_shape)
            h_y = y_batch.matmul(hadamard).view(y_shape)
            torch.cuda.synchronize()
            time_flag = time.time()
            out2 = quantize_forward_easy.quantize(h_x,h_y,scale_hx, scale_hy)
            torch.cuda.synchronize()
            time2 = time.time()
            if i>= 1:
                hadamard_time += time_flag - time1
                quantize_time += out2[3][0]
                pack_time += out2[3][1]
                gemm_time += out2[3][2]
                dequantize_time += out2[3][3]
                total_time += time2 - time1
        print("HQ cuda MM speed:")
        print("    Tflops is:", 1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        # print("    output is:", out2[0])
        print("total_time is:")
        print(total_time)
        print("gemm_time is:")
        print(gemm_time)
        # print("nz is:")
        # print(x.shape[1])
        hadamard_cuda_speed_tflops.append(1e-12 * mconfig.M * mconfig.K * mconfig.N * mconfig.testTurn * 2 / total_time)
        
def write_flops():
    with open("./image/flops.txt", "w") as f:
        for index in range(len(matrix_shape)):
            print("matrix shape is:", matrix_shape[index], file=f)
            print("    fp16 flops is:", python_ordgemm_flops[index], file=f)
            print("    HQ flops is:", hadamard_cuda_speed_tflops[index], file=f)
            print("    LssWeight flops is:", twolayer_cuda_speed_tflops[index], file=f)
            print("    LssInput flops is:", twolayerInput_cuda_speed_tflops[index], file=f)
            print("    Average flops is:", 3/(1/twolayerInput_cuda_speed_tflops[index] + 1/twolayer_cuda_speed_tflops[index] + 1/hadamard_cuda_speed_tflops[index]), file=f)
            print("", file=f)
            
            
def draw_picture_flops():
    plt.figure(figsize=(25, 20))
    bar_width = 0.17
    
    r1 = range(len(matrix_shape))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]
    r4 = [x + bar_width for x in r3]
    r5 = [x + bar_width for x in r4]
    
    plt.bar(r1, python_ordgemm_flops, width=bar_width, edgecolor='white', label='FP16')
    plt.bar(r2, 3/(1/np.array(twolayer_cuda_speed_tflops) + 1/np.array(twolayerInput_cuda_speed_tflops) + 1/np.array(hadamard_cuda_speed_tflops)), width=bar_width, edgecolor='white', label='INT4')
    plt.bar(r3, hadamard_cuda_speed_tflops, width=bar_width, edgecolor='white', label='HQ')
    plt.bar(r4, twolayer_cuda_speed_tflops, width=bar_width, edgecolor='white', label='LSSWeight')
    plt.bar(r5, twolayerInput_cuda_speed_tflops, width=bar_width, edgecolor='white', label='LSSAct')
    
    plt.xticks(r2, matrix_shape, rotation=30, fontsize=45)
    plt.yticks(fontsize=60)
    
    plt.legend(loc='upper left', fontsize=60)

    font = {'size' : 60}
    plt.xlabel("Matrix size (M,N,K)",font)
    plt.ylabel('Tflops', font)
    # plt.title('Comparison of FP16 MM, HQ and LSS operator',fontsize=60)
    
    plt.savefig('./image/plot_flops.pdf', bbox_inches='tight')
    plt.savefig('./image/plot_flops.svg', bbox_inches='tight', format='svg')
    
def draw_picture_full():
    plt.figure(figsize=(20, 20))
    area = plt.subplot2grid((11,11),(0,0),rowspan=11, colspan = 10)
    area.plot()
    bar_width = 0.6
    
    data = [np.array(cuda_quantize_time), cuda_leverage_time, np.array(cuda_sample_time), cuda_pack_time, cuda_gemm_time, cuda_dequantize_time, cuda_LSQ_time]
    labels = ["quantize", "leverage", "sample", "pack", "gemm", "dequantize", "LSQ"]
    colors = ['#ff7f0e', '#8c564b', '#e377c2', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
        
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
            plt.text(r1[i], tmp_y+y/2, text, color='white',size='40',horizontalalignment='center',verticalalignment='center')
            tmp_y += y
    
    plt.xticks(r1, matrix_shape, rotation=30, fontsize=45)
    plt.yticks(fontsize=60)

    # plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=45)

    plt.title("LSSWeight", fontdict={'size' : 60})
    plt.ylabel('Time ratio', fontdict={'size' : 60})
    plt.xlabel("Matrix size (M,N,K)", fontdict={'size' : 60})

    # plt.savefig('./image/LssWeight.pdf', bbox_inches='tight')
    plt.savefig('./image/LssWeight.svg', bbox_inches='tight', format='svg')
    
def draw_picture_full2():
    plt.figure(figsize=(20, 20))
    area = plt.subplot2grid((11,11),(0,0),rowspan=11, colspan = 10)
    area.plot()
    bar_width = 0.6
    
    data = [np.array(cuda_input_sample_time), cuda_input_pack_time, cuda_input_gemm_time, cuda_input_dequantize_time, cuda_input_LSQ_time]
    labels = ["sample", "pack", "gemm", "dequantize", "LSQ"]
    colors = ['#e377c2', '#2ca02c', '#d62728', '#9467bd', '#7f7f7f']
        
    r1 = range(len(matrix_shape))
    
    bottom_y = np.zeros(len(matrix_shape))
    data = np.array(data)
    sums = np.sum(data, axis=0)
    # print(sums)
    
    for index in range(len(data)):
        y = data[index] / sums
        plt.bar(r1, y, width=bar_width, edgecolor='white', label=labels[index], bottom=bottom_y, color=colors[index])
        bottom_y += y
        
    for i in range(data.shape[1]):
        tmp_y = 0
        for j in range(data.shape[0]):
            y = data[j][i] / sums[i]
            text = "%d%%" % (100 * y)
            plt.text(r1[i], tmp_y+y/2, text, color='white',size='36',horizontalalignment='center',verticalalignment='center')
            tmp_y += y
    
    plt.xticks(r1, matrix_shape, rotation=30, fontsize=45)
    plt.yticks(fontsize=60)

    # plt.legend(loc='upper left', bbox_to_anchor=(1,1), ncol=1, fontsize=45)

    plt.title("LSSAct", fontdict={'size' : 60})
    plt.ylabel('Time ratio', fontdict={'size' : 60})
    plt.xlabel("Matrix size (M,N,K)", fontdict={'size' : 60})

    # plt.savefig('./image/LSSInput.pdf', bbox_inches='tight')
    plt.savefig('./image/LSSInput.svg', bbox_inches='tight', format='svg')
    
if __name__=="__main__":
    # ,(15360,8704,10752), (),,(35840,35840,43520)
    # for (m,n,k) in [(25600,25600,32768),(28672,28672,34816),(30720,30720,46080),(32768,32768,38912)]:
    for (m,n,k) in [(4608,5120,6144),(5120,6144,8192),(6144,6144,9216),(7168,6656,8704),(8192,7680,9728),(15360,8704,10752)]:
        print("matrix multiplication of {M,N,K} = {%d, %d, %d}" % (m,n,k))
        mconfig.M = m
        mconfig.N = n
        mconfig.K = k
        matrix_shape.append((mconfig.M, mconfig.N, mconfig.K))
        test = PreconditionerTest()
        returnList = test.TwoLayerQuantizeWeight_cuda_speed(test.x)
        test.Gemm_ordinary_python(test.x, test.y)
        test.HadmardQuantize_cuda_speed(test.x.t().contiguous(), test.y.t().contiguous())
        test.TwoLayerQuantizeInput_cuda_speed(test.x, returnList)

    draw_picture_flops()
    write_flops()
    draw_picture_full()
    draw_picture_full2()