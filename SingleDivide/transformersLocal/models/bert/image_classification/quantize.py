from collections import namedtuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.nn.parameter import Parameter
from torch.autograd.function import InplaceFunction, Function
import matplotlib.pyplot as plt
from copy import copy, deepcopy
import time
import random
import quantize_forward_easy as qfe
import quantize_grad_weight_speed as qgw
import quantize_grad_input_speed as qgi
import shutil
# import quantize_grad as qg

try:
    from .preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, SingleDivideGWPreconditioner, SingleDivideGAPreconditioner,\
        TwoLayerWeightPreconditioner, LUQPreconditioner, HadamardMultiplier
    from .utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN, statistic, draw
    from .activation_quantizer_1 import SymLsqQuantizer, AsymLsqQuantizer, LsqStepSize, \
        act_quant_fn, weight_quant_fn
    from .special_quantize import special_quantize_grad_weight, special_quantize_grad_input, fake_quantize, special_quantize_grad_input_float, special_quantize_grad_weight_float, special_quantize_grad_weight_test
except:
    from preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, SingleDivideGWPreconditioner, SingleDivideGAPreconditioner,\
        TwoLayerWeightPreconditioner, LUQPreconditioner, HadamardMultiplier
    from utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN, statistic, draw
    from activation_quantizer_1 import SymLsqQuantizer, AsymLsqQuantizer, LsqStepSize, \
        act_quant_fn, weight_quant_fn
    from special_quantize import special_quantize_grad_weight, special_quantize_grad_input, fake_quantize, special_quantize_grad_input_float, special_quantize_grad_weight_float, special_quantize_grad_weight_test
import os
import matplotlib.pyplot as plt


class QuantizationConfig:
    def __init__(self):
        self.quantize_activation = True
        self.quantize_weights = True
        self.quantize_gradient = True
        self.activation_num_bits = 8
        self.weight_num_bits = 8
        self.bias_num_bits = 16
        self.backward_num_bits = 8
        self.bweight_num_bits = 8
        self.backward_persample = False
        self.biased = False
        self.grads = None
        self.acts = None
        self.hadamard = False
        self.dynamic = True
        self.bmm = True
        self.biprecision = True
        self.twolayers_gradweight = False
        self.twolayers_gradinputt = False
        self.luq = False
        self.forward_method = 'PTQ'
        self.clip_value = 0
        self.choice = None

        self.weight_quant_method = 'LSQ'
        self.input_quant_method = ''
        self.learnable_step_size = True
        self.learnable_hadamard = True
        self.lsq_layerwise_input = 'layer'
        self.lsq_layerwise_weight = 'layer'
        self.retain_large_value = False
        self.quantize_large_value = False
        self.draw_value = False

        self.track_step_size = False
        self.fp16 = False
        self.forward = {}
        self.backward = {}
        self.hadamard_time = 0
        self.special_layer_time = 0
        self.abs_time = 0
        self.forward_all = 0
        self.backward_all = 0
        self.forward_test = 0
        self.forward_calculate = 0

        self.group_search = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def activation_preconditioner(self):
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)

    def weight_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.weight_num_bits)

    def activation_gradient_preconditioner(self, special=False, single=False):
        if self.luq:
            return lambda x: LUQPreconditioner(x, self.backward_num_bits)

        if special:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

        if single:
            return lambda x: SingleDivideGAPreconditioner(x, self.backward_num_bits)

        if self.twolayers_gradinputt:
            return lambda x: TwoLayerWeightPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self, special=False, single=False):
        if self.luq:
            return lambda x: LUQPreconditioner(x, self.bweight_num_bits)

        if special:
            return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)

        if single:
            return lambda x: SingleDivideGWPreconditioner(x, self.bweight_num_bits)

        if self.twolayers_gradweight:
            return lambda x: TwoLayerWeightPreconditioner(x, self.bweight_num_bits)


qconfig = QuantizationConfig()

QParams = namedtuple('QParams', ['range', 'zero_point', 'num_bits'])

_DEFAULT_FLATTEN = (1, -1)
_DEFAULT_FLATTEN_GRAD = (0, -1)


class UniformQuantize(InplaceFunction):

    @staticmethod
    def forward(ctx, input, Preconditioner, stochastic=False, inplace=False):
        ctx.inplace = inplace

        if ctx.inplace:
            ctx.mark_dirty(input)
            output = input
        else:
            output = input.clone()

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print('---')
        #     print(input.view(-1)[:10], input.min(), input.max())
        with torch.no_grad():
            preconditioner = Preconditioner(output)
            output = preconditioner.forward()

            if stochastic:
                noise = output.new(output.shape).uniform_(-0.5, 0.5)
                output.add_(noise)
                # print("quantize 2", output)
                if qconfig.luq:
                    log_bias = math.log(4 / 3) - 1 / 2
                    output.add_(torch.ones_like(output) * log_bias)
                    # print("quantize 3", output)
            # quantize
            output.clamp_(preconditioner.Qn, preconditioner.Qp).round_()

            output = preconditioner.inverse(output)

        # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        #     print(output.view(-1)[:10])
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None


def quantize(x, Preconditioner, stochastic=False, inplace=False):
    return UniformQuantize().apply(x, Preconditioner, stochastic, inplace)

def time_forward(time_vector):
    key_list = ["quantize", "pack", "gemm", "dequantize"]
    for keyindex in range(len(key_list)):
        keyname = key_list[keyindex]
        if keyname not in qconfig.forward.keys():
            qconfig.forward[keyname] = time_vector[keyindex]
        else:
            qconfig.forward[keyname] += time_vector[keyindex]
            
def time_backward(time_vector):
    key_list = ["leverage", "sample1", "sample2", "quantize", "contiguous", "pack", "fp16gemm", "int4gemm", "dequantize", "LSQ"]
    for keyindex in range(len(key_list)):
        keyname = key_list[keyindex]
        if keyname not in qconfig.backward.keys():
            qconfig.backward[keyname] = time_vector[keyindex]
        else:
            qconfig.backward[keyname] += time_vector[keyindex]
            
class linear_act_test(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        # out[1] is q_input_flatten, out[2] is q_weight
        # h_input_flatten ~ q_input_flatten * scale_input.half()
        # h_weight ~ q_weight * scale_weight.half()
        # IPython.embed()
        if qconfig.fp16:
            # assert input.dtype == torch.float16
            # assert weight.dtype == torch.float16
            if input.dtype != torch.float16:
                input = input.half()
            if weight.dtype != torch.float16:
                weight = weight.half()
            assert bias.dtype == torch.float16
        input_flatten = input.reshape(-1, input.shape[-1])
        input_shape = input.shape
        ctx.saved = input_flatten, weight, bias, input_shape
        out_shape = list(input.shape)
        out_shape[-1] = weight.shape[0]
        output = input_flatten.mm(weight.t()).reshape(out_shape)
        if bias is not None:
            output = output + bias
        # else:
        #     return output
        return output

    @staticmethod
    def backward(ctx, grad_output):
        if qconfig.fp16:
            assert grad_output.dtype == torch.float16
        # print(grad_output.shape)

        # input, weight, bias = ctx.saved
        input_flatten, weight, bias, input_shape = ctx.saved

        # C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten = grad_output.reshape(-1, C_out)
        # up corresponds to input cuda, down corresponds to weight cuda
        # IPython.embed()
        grad_input_flatten = grad_output_flatten.mm(weight)
        grad_weight = grad_output_flatten.t().mm(input_flatten)
            
        grad_bias_start = time.time()
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        if "grad_bias" not in qconfig.forward.keys():
            qconfig.backward["grad_bias"] = time.time() - grad_bias_start
        else:
            qconfig.backward["grad_bias"] += time.time() - grad_bias_start
        # grad_input_transform = grad_input.reshape(input.size())
        grad_input = grad_input_flatten.reshape(input_shape)
        
        if qconfig.fp16:
            assert grad_input.dtype == torch.float16
            assert grad_weight.dtype == torch.float16
            assert grad_bias.dtype == torch.float16

        # print("grad_input_transform", grad_input_transform)
        return grad_input, grad_weight, grad_bias

class linear_act_python(Function):
    @staticmethod
    def forward(ctx, h_input, h_weight, scale_input, scale_weight, bias, layer_name):
        #### torch.cuda.synchronize()
        # start = time.time()
        # if qconfig.fp16:
        #     assert h_input.dtype == torch.float16
        #     assert h_weight.dtype == torch.float16
        #     assert scale_input.dtype == torch.float16
        #     assert scale_weight.dtype == torch.float16
        #     assert bias.dtype == torch.float16
            
        input_shape = h_input.shape
        C_in = h_input.shape[-1]
        
        h_input_flatten = h_input.reshape(-1,C_in)
        # h_input_flatten = h_input.reshape(-1,C_in).half()
        # h_weight = h_weight.half()
        
        out_shape = list(h_input.shape)
        out_shape[-1] = h_weight.shape[0]
        # IPython.embed()
        #### torch.cuda.synchronize()
        # start2 = time.time()
        out = qfe.quantize(h_input_flatten, h_weight, scale_input, scale_weight)
        #### torch.cuda.synchronize()
        # qconfig.forward_calculate += time.time() - start2
        # time_forward(out[3])
        output = out[0].reshape(out_shape)
        
        # out[1] is q_input_flatten, out[2] is q_weight
        # h_input_flatten ~ q_input_flatten * scale_input.half()
        # h_weight ~ q_weight * scale_weight.half()
        ctx.saved = out[1], out[2], out[3], out[4], scale_input, scale_weight, bias, input_shape, layer_name
        
        if bias is not None:
            output = output + bias
        # else:
        #     return output
        
        #### torch.cuda.synchronize()
        # qconfig.forward_test += time.time() - start
        return output

    @staticmethod
    def backward(ctx, grad_output):
        #### torch.cuda.synchronize()
        # backward_start = time.time()
        # if qconfig.fp16:
        #     assert grad_output.dtype == torch.float16
        # print(grad_output.shape)

        # input, weight, bias = ctx.saved
        q_input_flatten, q_weight, lsq_input_flatten, lsq_weight, scale_input, scale_weight, bias, input_shape, layer_name = ctx.saved

        # C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten = grad_output.reshape(-1, C_out)
        
        flag_weight = (q_input_flatten.shape[1] % 4 != 0)
        if flag_weight:
        # if True:
            if qconfig.fp16:
                grad_weight, grad_scale_weight, first_transform, second_transform, x1_len, x2_len, scale1  = special_quantize_grad_weight(grad_output_flatten, q_input_flatten, scale_input, 4, lsq_weight)
            else:
                grad_weight, grad_scale_weight, first_transform, second_transform, x1_len, x2_len, scale1 = special_quantize_grad_weight_float(grad_output_flatten, q_input_flatten, scale_input, 4, lsq_weight)
        else:
            weight_out = qgw.quantize(grad_output_flatten, 4, q_input_flatten, scale_input, lsq_weight)
            grad_weight, grad_scale_weight = weight_out[0], weight_out[1]
            first_transform, second_transform, x1_len, x2_len, scale1 = weight_out[-5], weight_out[-4], weight_out[-3], weight_out[-2], weight_out[-1]
            
        flag_input = (grad_output_flatten.shape[1] % 32 != 0) or (q_weight.shape[1] % 4 != 0)
        if flag_input or flag_weight:
        # if True:
            if qconfig.fp16:
                grad_input_flatten, grad_scale_input = special_quantize_grad_input(grad_output_flatten, q_weight, scale_weight, 4, lsq_input_flatten)
            else:
                grad_input_flatten, grad_scale_input = special_quantize_grad_input_float(grad_output_flatten, q_weight, scale_weight, 4, lsq_input_flatten)
                # grad_input_flatten, grad_scale_input = special_quantize_grad_test_float(grad_output_flatten, q_weight, scale_weight, 4, lsq_input_flatten, weight_out[-5], weight_out[-4], weight_out[-3], weight_out[-2], weight_out[-1])
        else:
            activation_out = qgi.quantize(grad_output_flatten, 4, q_weight, scale_weight, lsq_input_flatten, first_transform, second_transform, x1_len, x2_len, scale1)
            grad_input_flatten, grad_scale_input = activation_out[0], activation_out[1]

        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        # if "grad_bias" not in qconfig.forward.keys():
        #     qconfig.backward["grad_bias"] = time.time() - grad_bias_start
        # else:
        #     qconfig.backward["grad_bias"] += time.time() - grad_bias_start
            
        # grad_input_transform = grad_input.reshape(input.size())
        grad_input = grad_input_flatten.reshape(input_shape)
        
        # if qconfig.fp16:
        #     assert grad_input.dtype == torch.float16
        #     assert grad_weight.dtype == torch.float16
        #     assert grad_scale_input.dtype == torch.float16
        #     assert grad_scale_weight.dtype == torch.float16
        #     assert grad_bias.dtype == torch.float16

        #### torch.cuda.synchronize()
        # qconfig.backward_all += time.time() - backward_start
        # print("grad_input_transform", grad_input_transform)
        return grad_input, grad_weight, grad_scale_input, grad_scale_weight, grad_bias, None

class identity_act(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.saved = input
        return input

    @staticmethod
    def backward(ctx, grad_output):
        # torch.set_printoptions(profile="full", linewidth=160)
        grad_output_weight_conditioner = quantize(grad_output,
                                                  qconfig.weight_gradient_preconditioner(special=True),
                                                  stochastic=True)
        input = ctx.saved
        C_out = grad_output.shape[-1]

        grad_output_flatten_weight = grad_output_weight_conditioner.reshape(-1, C_out)

        grad_input = grad_output_flatten_weight
        grad_input_transform = grad_input.reshape(input.size())
        return grad_input_transform


class QuantMeasure(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, inplace=False, stochastic=False):
        super(QuantMeasure, self).__init__()
        self.stochastic = stochastic
        self.inplace = inplace

    def forward(self, input):
        q_input = quantize(input, qconfig.activation_preconditioner(),
                           stochastic=self.stochastic, inplace=self.inplace)
        return q_input
    
    
class QLinear(nn.Linear):
    """docstring for QConv2d."""

    def __init__(self, in_features, out_features, bias=True, name=''):
        super(QLinear, self).__init__(in_features, out_features, bias)
        # self.quantize_input = QuantMeasure()
        self.name = name

        self.create_H()
        self.first_pass = False
        self.initialize_weight = False
        self.initialize_input = False
        self.best_group = 0
        if not qconfig.bmm and not qconfig.dynamic:
            self.hadamard_static = self.T[32]
            
        self.active_track, self.weight_track, self.iter_track = [], [], []

        self.scale_input = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.scale_weight = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        
    def create_H(self):
        self.T = {}

        size = 1
        if qconfig.fp16:
            H = torch.ones(1, 1).cuda().half()
        else:
            H = torch.ones(1, 1).cuda()
        self.T[1] = H

        for i in range(10):
            H = torch.cat((torch.cat([H, H], 1),
                           torch.cat([H, -H], 1)), 0) / math.sqrt(2)
            size *= 2
            self.T[size] = H
            
    def draw_clip_value(self):
        current_time = self.name_draw_clip_value

        plt.figure()
        plt.title("{}\n{}".format(self.active_track[0], self.active_track[-1]))
        plt.plot(self.iter_track, self.active_track)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

        if not os.path.exists("plt4/step_size_linear/input/{}".format(len(self.iter_track))):
            print("Directory {} created".format("plt4/step_size_linear/input/{}".format(len(self.iter_track))))

        os.makedirs("plt4/step_size_linear/input/{}".format(len(self.iter_track)), exist_ok=True)
        plt.savefig("plt4/step_size_linear/input/{}/{}.png".format(len(self.iter_track), current_time))
        plt.close()

        plt.figure()
        plt.title("{}\n{}".format(self.weight_track[0], self.weight_track[-1]))
        plt.plot(self.iter_track, self.weight_track)
        plt.gca().get_yaxis().get_major_formatter().set_useOffset(False)

        if not os.path.exists("plt4/step_size_linear/weight/{}".format(len(self.iter_track))):
            print("Directory {} created".format("plt4/step_size_linear/weight/{}".format(len(self.iter_track))))

        os.makedirs("plt4/step_size_linear/weight/{}".format(len(self.iter_track)), exist_ok=True)
        plt.savefig("plt4/step_size_linear/weight/{}/{}.png".format(len(self.iter_track), current_time))
        plt.close()
            
    def forward(self, input):
        # if qconfig.fp16:
        #     assert(input.dtype == torch.float16)
        # print("input:",input.is_contiguous())
        # assert(input.is_contiguous())
        #### torch.cuda.synchronize()
        # abs_start = time.time()
        
        if self.first_pass:
            self.scale_input.data = self.scale_input.data.abs()
            self.scale_weight.data = self.scale_weight.data.abs()
            
        #### torch.cuda.synchronize()
        # if "abs" not in qconfig.forward.keys():
        #     qconfig.forward["abs"] = time.time() - abs_start
        # else:
        #     qconfig.forward["abs"] += time.time() - abs_start

        if not self.first_pass:
            print("Actually Using QLinear!")
            self.first_pass = True

            if qconfig.hadamard and qconfig.dynamic:
                raw_input = deepcopy(input.detach())
                best_hadamard_group, min_hadamard_quantize_error = 0, 1e10
                for H_search in qconfig.group_search:
                    if input.shape[-1] % H_search != 0:
                        continue
                    H_search_input = raw_input
                    # assert(H_search_input.is_contiguous())
                    
                    dim = H_search_input.shape[-1]
                    
                    if qconfig.bmm:
                        hadamard = self.T[H_search].repeat(dim // H_search, 1, 1)
                    else:
                        hadamard = self.T[H_search]
                    
                    input_shape = input.shape
                    weight_shape = self.weight.shape
                    # H_search_input_flatten = H_search_input.reshape(-1, H_search_input.shape[-1])
                    # assert(H_search_input_flatten.is_contiguous())
                    
                    if qconfig.bmm:
                        H_search_input = torch.bmm(H_search_input.reshape(-1, H_search_input.shape[-1]).reshape(-1, dim // H_search, H_search).transpose(0,1), hadamard).transpose(0,1).reshape(input_shape)
                    else:
                        H_search_input = H_search_input.reshape(-1, H_search_input.shape[-1]).reshape(-1, H_search).matmul(hadamard).reshape(input_shape)
                    
                    # assert(H_search_input.is_contiguous())
                    # weight_in = self.HM_w(self.weight)
                    
                    if qconfig.bmm:
                        weight_in = torch.bmm(self.weight.reshape(-1, dim // H_search, H_search).transpose(0,1), hadamard).transpose(0,1).reshape(weight_shape)
                    else:
                        weight_in = self.weight.reshape(-1, H_search).matmul(hadamard).reshape(weight_shape)
                    
                    scale_input = 2 * H_search_input.abs().mean() / math.sqrt(7) + 1e-10
                    scale_weight = 2 * weight_in.abs().mean() / math.sqrt(7) + 1e-10
                    # print(H_search_input.shape, weight_in.shape)
                    qinput, qweight = fake_quantize(H_search_input, weight_in, scale_input, scale_weight)

                    # qinput = self.HMinv(qinput_flatten.reshape(input_shape).float())
                    # qweight = self.HMinv_w(qweight.float())
                    if qconfig.bmm:
                        qinput = torch.bmm(qinput.reshape(-1, dim // H_search, H_search).transpose(0,1), hadamard).transpose(0,1).reshape(input_shape)
                        qweight = torch.bmm(qweight.reshape(-1, dim // H_search, H_search).transpose(0,1), hadamard).transpose(0,1).reshape(weight_shape)
                    else:
                        qinput = qinput.reshape(-1, qinput.shape[-1]).reshape(-1, H_search).matmul(hadamard).reshape(input_shape)
                        qweight = qweight.reshape(-1, H_search).matmul(hadamard).reshape(weight_shape)

                    Q_error_a, Q_error_w = (qinput - raw_input).norm(), (qweight - self.weight).norm()
                    Q_error = Q_error_a * Q_error_w
                    if Q_error < min_hadamard_quantize_error:
                        min_hadamard_quantize_error = Q_error
                        best_hadamard_group = H_search

                    # print(f"when group is {H_search} the error is A: {Q_error_a} and W: {Q_error_w}")

                self.best_group = best_hadamard_group
                if qconfig.bmm:
                    self.hadamard_dynamic = self.T[self.best_group].repeat(dim // self.best_group, 1, 1)
                else:
                    self.hadamard_dynamic = self.T[self.best_group]

                print(f"At layer   {self.name}   The best group is {best_hadamard_group}")

        #### torch.cuda.synchronize()
        # hadamard_start = time.time()
        dim = input.shape[-1]
        if qconfig.bmm and not qconfig.dynamic:
            self.hadamard_static = self.T[32].repeat(dim // 32, 1, 1)
        
        if qconfig.hadamard:
            # h_input = self.HM(input)
            input_flatten = input.reshape(-1, input.shape[-1])
            if qconfig.dynamic:
                if qconfig.bmm:
                    if self.best_group != 1:
                        h_input = torch.bmm(input_flatten.reshape(-1, dim // self.best_group, self.best_group).transpose(0,1), self.hadamard_dynamic).transpose(0,1).reshape(input.shape)
                    else:
                        h_input = input
                else:
                    h_input = input_flatten.reshape(-1, self.best_group).matmul(self.hadamard_dynamic).reshape(input.shape)
            else:
                if qconfig.bmm:
                    if self.best_group != 1:
                        h_input = torch.bmm(input_flatten.reshape(-1, dim // 32, 32).transpose(0,1), self.hadamard_static).transpose(0,1).reshape(input.shape)
                    else:
                        h_input = input
                else:
                    h_input = input_flatten.reshape(-1, 32).matmul(self.hadamard_static).reshape(input.shape) 
        # print("h_input:",h_input.is_contiguous())
        
        if qconfig.hadamard:
            # h_weight = self.HM_w(self.weight)
            if qconfig.dynamic:
                if qconfig.bmm:
                    if self.best_group != 1:
                        h_weight = torch.bmm(self.weight.reshape(-1, dim // self.best_group, self.best_group).transpose(0,1), self.hadamard_dynamic).transpose(0,1).reshape(self.weight.shape)
                    else:
                        h_weight = self.weight
                else:
                    h_weight = self.weight.reshape(-1, self.best_group).matmul(self.hadamard_dynamic).reshape(self.weight.shape)
            else:
                if qconfig.bmm:
                    if self.best_group != 1:
                        h_weight = torch.bmm(self.weight.reshape(-1, dim // 32, 32).transpose(0,1), self.hadamard_static).transpose(0,1).reshape(self.weight.shape)
                    else:
                        h_weight = self.weight
                else:
                    h_weight = self.weight.reshape(-1, 32).matmul(self.hadamard_static).reshape(self.weight.shape)
        else:
            h_weight = self.weight
            
        #### torch.cuda.synchronize()
        # if "hadamard" not in qconfig.forward.keys():
        #     qconfig.forward["hadamard"] = time.time() - hadamard_start
        # else:
        #     qconfig.forward["hadamard"] += time.time() - hadamard_start

        if not self.initialize_input:
            # IPython.embed()
            # self.scale_input = nn.Parameter(2 * h_input.abs().mean() / math.sqrt(7) + 1e-10, requires_grad=True)
            self.scale_input.data.copy_(2 * h_input.abs().mean() / math.sqrt(7) + 1e-10)
            print("init", self.scale_input.min())
            self.initialize_input = True
            
        if not self.initialize_weight:
            # self.scale_weight = nn.Parameter(2 * h_weight.abs().mean() / math.sqrt(7) + 1e-10, requires_grad=True)
            self.scale_weight.data.copy_(2 * h_weight.abs().mean() / math.sqrt(7) + 1e-10)
            print("init", self.scale_weight.min())
            self.initialize_weight = True
        
        # bias_start = time.time()
        qbias = self.bias
        if qconfig.fp16:
            qbias = qbias.half()
        #### torch.cuda.synchronize()
        # if "bias" not in qconfig.forward.keys():
        #     qconfig.forward["bias"] = time.time() - bias_start
        # else:
        #     qconfig.forward["bias"] += time.time() - bias_start
        

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_act_python.apply(h_input, h_weight, self.scale_input, self.scale_weight, qbias, self.name_draw_clip_value)
            # output = linear_act_test.apply(input, self.weight, qbias)

        if qconfig.input_quant_method == 'lsq':
            self.active_track.append(self.scale_input.cpu().detach().numpy())
        if qconfig.weight_quant_method == 'lsq':
            self.weight_track.append(self.scale_weight.cpu().detach().numpy())
        self.iter_track.append(len(self.iter_track))

        # if qconfig.input_quant_method == 'lsq':
        #     if len(self.iter_track) % 200 == 50:
        #         self.draw_clip_value()

        # if self.first_pass:
        #     #### torch.cuda.synchronize()
        #     qconfig.forward_all += time.time() - abs_start
        return output


# Todo:暂定为QEmbedding之后的线性补充层
# 此处输入为embedding层的weight，需要按照weight的方式进行量化
class QIdentity(nn.Identity):
    def __init__(self, name=''):
        self.name = name
        super(QIdentity, self).__init__()

        self.quantize_input = QuantMeasure()
        self.first_pass = False
        if qconfig.lsq_layerwise_weight != "row":
            self._build_embed_clip_val(qconfig.weight_quant_method, init_val=qconfig.clip_value)


    def _build_embed_clip_val(self, quant_method, init_val, example=None):
        # print(quant_method, init_val, qconfig.change_type, qconfig.lsq_layerwise)
        # print('!'*1000)
        if quant_method == 'uniform':
            self.register_buffer('embed_clip_val', torch.tensor([-init_val, init_val]))
            self.embed_clip_val = nn.Parameter(self.embed_clip_val)
        elif quant_method == 'lsq':
            # TODO: for now we abuse the name for consistent reference in learner.
            # assert learnable, 'LSQ must use learnable step size!'
            if qconfig.lsq_layerwise_weight == 'layer':
                self.embed_clip_val = LsqStepSize(
                    torch.tensor(1.0, requires_grad=True))  # stepsize will be initialized in the first quantization
            elif qconfig.lsq_layerwise_weight == 'row':

                self.embed_clip_val = LsqStepSize(
                    torch.ones_like(example[:, :, 0]).clone().detach().requires_grad_(
                        True))  # stepsize will be initialized in the first quantization
            elif qconfig.lsq_layerwise_weight == 'column':
                if example is not None:
                    self.embed_clip_val = LsqStepSize(
                        torch.ones_like(example[0, 0, :]).clone().detach().requires_grad_(
                            True))  # stepsize will be initialized in the first quantization\
                else:
                    self.embed_clip_val = LsqStepSize(
                        torch.ones(768).clone().detach().requires_grad_(
                            True))  # stepsize will be initialized in the first quantization

        else:
            self.register_buffer('embed_clip_val', None)

    def draw(self, x, s='', fix=''):
        x = x.reshape(-1).cpu().numpy()

        plt.figure(figsize=(15, 15))
        num_bins = 1024
        n, bins, patches = plt.hist(x, num_bins, density=1, color='green')

        plt.ylabel('Y-Axis')
        plt.xlim([x.min() * 1.1, x.max() * 1.1])
        plt.ylim([0, n.max() * 1.1])
        plt.title("ratio={}\nmin={}\nmax={}".format(np.abs((x.min() / x.max())), x.min(), x.max()),
                  fontweight="bold")

        os.makedirs("plt/{}/{}/{}".format(self.name, self.time_time, s), exist_ok=True)
        plt.savefig(
            'plt/{}/{}/{}/{}.png'.format(self.name, self.time_time, s, fix))
        plt.close()
        
    def forward(self, input):
        self.time_time = time.time_ns()

        if self.first_pass:
            if qconfig.weight_quant_method == 'lsq':
                self.embed_clip_val.change_abs()
        # print(input.shape)
        if not self.first_pass:
            print("Actually Using QIdentity!")
            self.first_pass = True

            if qconfig.hadamard:
                raw_input = deepcopy(input.detach())
                best_hadamard_group, min_hadamard_quantize_error = 0, 1e10
                for H_search in qconfig.group_search:
                    if input.shape[-1] % H_search != 0:
                        continue
                    H_search_input = raw_input
                    self.HM = HadamardMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                    learnable=qconfig.learnable_hadamard)
                    self.HMinv = HadamardMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                              learnable=qconfig.learnable_hadamard)
                    if qconfig.draw_value:
                        self.draw(H_search_input.detach(), f"{H_search}_input_search_hadamard", "1 origin")
                    if qconfig.hadamard:
                        H_search_input = self.HM(H_search_input)
                    if qconfig.draw_value:
                        self.draw(H_search_input.detach(), f"{H_search}_input_search_hadamard", "2 hadamard")
                    if qconfig.weight_quant_method == 'ptq':
                        qinput = quantize(H_search_input, qconfig.weight_preconditioner())
                    else:
                        self.embed_clip_val.initialized = False
                        qinput = weight_quant_fn(H_search_input, self.embed_clip_val, num_bits=qconfig.weight_num_bits,
                                                 symmetric=True,
                                                 quant_method=qconfig.weight_quant_method,
                                                 layerwise=qconfig.lsq_layerwise_weight,
                                                 learnable=qconfig.learnable_step_size)
                        self.embed_clip_val.initialized = False
                    if qconfig.draw_value:
                        self.draw(qinput.detach(), f"{H_search}_input_search_hadamard", "3 quantize")
                    if qconfig.hadamard:
                        qinput = self.HMinv(qinput)
                    if qconfig.draw_value:
                        self.draw(qinput.detach(), f"{H_search}_input_search_hadamard", "4 hadamard inv")

                    Q_error = (qinput - raw_input).norm()
                    if Q_error < min_hadamard_quantize_error:
                        min_hadamard_quantize_error = Q_error
                        best_hadamard_group = H_search

                    # print(f"when group is {H_search} the error is {Q_error}")

                self.HM = HadamardMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                learnable=qconfig.learnable_hadamard)
                self.HMinv = HadamardMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                          learnable=qconfig.learnable_hadamard)

                print(f"At layer   {self.name}   The best group is {best_hadamard_group}")

        if qconfig.draw_value:
            self.draw(input.detach(), "input", "1 origin")

        if qconfig.hadamard:
            input = self.HM(input)

        if qconfig.draw_value:
            self.draw(input.detach(), "input", "2 hadamard")

        if qconfig.weight_quant_method == 'ptq':
            qinput = quantize(input, qconfig.weight_preconditioner())
        else:
            qinput = weight_quant_fn(input, self.embed_clip_val, num_bits=qconfig.weight_num_bits, symmetric=True,
                                     quant_method=qconfig.weight_quant_method, layerwise=qconfig.lsq_layerwise_weight,
                                     learnable=qconfig.learnable_step_size)

        if qconfig.draw_value:
            self.draw(qinput.detach(), "input", "3 quantize")

        if qconfig.hadamard:
            qinput = self.HMinv(qinput)

        if qconfig.draw_value:
            self.draw(qinput.detach(), "input", "4 hadamard inv")

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = qinput
        else:
            output = identity_act.apply(qinput)

        # exit(0)

        return output
