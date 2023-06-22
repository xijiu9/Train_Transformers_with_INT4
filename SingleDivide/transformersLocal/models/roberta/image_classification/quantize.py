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

try:
    from .preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, ScalarPreconditionerActPerFeature,\
        lsq_per_tensor, lsq_plus, \
        TwoLayerWeightPreconditioner, LUQPreconditioner, HouseholderMultiplier, HouseholderInverseMultiplier
    from .utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN
    from .activation_quantizer_1 import SymLsqQuantizer, AsymLsqQuantizer, LsqStepSize, \
        act_quant_fn, weight_quant_fn
except:
    from preconditioner import ScalarPreconditioner, ScalarPreconditionerAct, ScalarPreconditionerActPerFeature,\
        lsq_per_tensor, lsq_plus, \
        TwoLayerWeightPreconditioner, LUQPreconditioner, HouseholderMultiplier
    from utils import twolayer_linearsample_weight, twolayer_linearsample_input, checkNAN
    from activation_quantizer_1 import SymLsqQuantizer, AsymLsqQuantizer, LsqStepSize, \
        act_quant_fn, weight_quant_fn
import IPython
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
        self.householder = False
        self.householder_weight = False
        self.perfeature = False
        self.hadamard_group = 0
        self.biprecision = True
        self.twolayers_gradweight = False
        self.twolayers_gradinputt = False
        self.luq = False
        self.forward_method = 'PTQ'
        self.cutood = None
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
        self.real_quantize = False

        self.group_search = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    def activation_preconditioner(self):
        if self.perfeature:
            return lambda x: ScalarPreconditionerActPerFeature(x, self.activation_num_bits)
        # return lambda x: ForwardPreconditioner(x, self.activation_num_bits)
        return lambda x: ScalarPreconditionerAct(x, self.activation_num_bits)
        # return lambda x: ScalarPreconditioner(x, 16)

    def weight_preconditioner(self, special_bit=0):
        if special_bit == 0:
            return lambda x: ScalarPreconditioner(x, self.weight_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, special_bit)
        # return lambda x: ForwardPreconditioner(x, self.weight_num_bits)
        # return lambda x: DiagonalPreconditioner(x, self.weight_num_bits)

    def bias_preconditioner(self):
        return lambda x: ScalarPreconditioner(x, self.bias_num_bits)

    def activation_gradient_preconditioner(self, special=False):
        if self.luq:
            return lambda x: LUQPreconditioner(x, self.backward_num_bits)
        if self.twolayers_gradinputt and not special:
            return lambda x: TwoLayerWeightPreconditioner(x, self.backward_num_bits)
        else:
            return lambda x: ScalarPreconditioner(x, self.backward_num_bits)

    def weight_gradient_preconditioner(self, special=False):
        if self.luq:
            return lambda x: LUQPreconditioner(x, self.bweight_num_bits)
        if self.twolayers_gradweight and not special:
            return lambda x: TwoLayerWeightPreconditioner(x, self.bweight_num_bits)
        return lambda x: ScalarPreconditioner(x, self.bweight_num_bits)


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
            output.clamp_(0.0, preconditioner.num_bins).round_()

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


class linear_act(Function):
    @staticmethod
    def forward(ctx, input, weight, bias):
        ctx.saved = input, weight, bias
        return F.linear(input, weight, bias)

    @staticmethod
    def backward(ctx, grad_output):
        # print("grad output", grad_output)
        checkNAN(grad_output, "grad output")
        # torch.set_printoptions(profile="full", linewidth=160)
        # print("grad output", grad_output.shape)
        input, weight, bias = ctx.saved
        grad_output_weight_conditioner = quantize(grad_output,
                                                  qconfig.weight_gradient_preconditioner(),
                                                  stochastic=True)

        special_flag = (weight.shape[0] < 5)

        if special_flag:
            grad_output_active_conditioner = quantize(grad_output,
                                                      qconfig.activation_gradient_preconditioner(special=special_flag),
                                                      stochastic=True)
        else:
            # grad_output_active_conditioner = quantize(grad_output.transpose(-2, -1), qconfig.activation_gradient_preconditioner(),
            #                                           stochastic=True).transpose(-2, -1)
            grad_output_active_conditioner = quantize(grad_output, qconfig.activation_gradient_preconditioner(),
                                                      stochastic=True)

        # print("shape of input is:", input.size())
        # print("shape of grad_output is:", grad_output.size())
        C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten = grad_output.reshape(-1, C_out)
        grad_output_flatten_weight = grad_output_weight_conditioner.reshape(-1, C_out)
        # if qconfig.twolayers_gradinputt and not special_flag:
        #     grad_output_flatten_active = grad_output_active_conditioner.reshape(-1, 2 * C_out)
        # else:
        #     grad_output_flatten_active = grad_output_active_conditioner.reshape(-1, C_out)
        grad_output_flatten_active = grad_output_active_conditioner.reshape(-1, C_out)

        # print(grad_output_flatten_active.shape, grad_output_flatten_weight.shape, input.shape, special_flag)
        input_flatten = input.reshape(-1, C_in)

        if qconfig.twolayers_gradweight:
            if qconfig.grads is not None:
                qconfig.grads[0].append(grad_output_flatten_weight)
                qconfig.grads[1].append(input_flatten)
                print("save grad")

            m1, m2 = twolayer_linearsample_weight(grad_output_flatten_weight, input_flatten)
            grad_weight = m1.t().mm(m2)
        else:
            # print("weight", grad_output_flatten_weight.shape, input_flatten.shape)
            grad_weight = grad_output_flatten_weight.t().mm(input_flatten)

        if qconfig.twolayers_gradinputt:

            if special_flag:
                grad_input = grad_output_flatten_active.mm(weight)
            else:
                I = torch.eye(grad_output_flatten_active.shape[0] // 2, device="cuda")
                grad_input, _ = twolayer_linearsample_input(grad_output_flatten_active, I)

                checkNAN(grad_input, "grad input before")
                grad_input = grad_input.mm(weight)
                checkNAN(grad_input, "grad input after")
                # print(grad_output_flatten_active.shape)
                if qconfig.grads is not None:
                    qconfig.grads[2].append(grad_output_flatten_active)
                    qconfig.grads[3].append(I)
        else:
            # print("input", grad_output_flatten_active.shape, weight.shape)
            grad_input = grad_output_flatten_active.mm(weight)
        if bias is not None:
            grad_bias = grad_output_flatten.sum(0)
        else:
            grad_bias = None
        grad_input_transform = grad_input.reshape(input.size())
        # print("shape of grad_input is:", grad_input.size())
        # print("shape of grad_weight is:", grad_weight.size())
        # print("shape of grad_bias is:", grad_bias.size())
        checkNAN(grad_input_transform, "grad input transform")
        # print("grad_input_transform", grad_input_transform)
        return grad_input_transform, grad_weight, grad_bias


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
        # print("shape of input is:", input.size())
        # print("shape of grad_output is:", grad_output.size())
        # C_in = input.shape[-1]
        C_out = grad_output.shape[-1]

        grad_output_flatten_weight = grad_output_weight_conditioner.reshape(-1, C_out)

        grad_input = grad_output_flatten_weight
        grad_input_transform = grad_input.reshape(input.size())
        # print("shape of grad_input is:", grad_input.size())
        # print("shape of grad_weight is:", grad_weight.size())
        # print("shape of grad_bias is:", grad_bias.size())
        return grad_input_transform


class UniformQuantizeSawb(InplaceFunction):

    @staticmethod
    def forward(ctx, input, c1, c2, Qp, Qn):
        output = input.clone()

        with torch.no_grad():
            clip = (c1 * torch.sqrt(torch.mean(input ** 2))) + (c2 * torch.mean(input.abs()))
            scale = 2 * clip / (Qp - Qn)
            output.div_(scale)
            output.clamp_(Qn, Qp).round_()
            output.mul_(scale)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # straight-through estimator
        grad_input = grad_output
        return grad_input, None, None, None, None


def get_sawb_coefficients(bits):
    bits = int(bits)
    coefficient_dict = {1: [0., 1.], 2: [3.19, -2.14], 3: [7.40, -6.66], 4: [11.86, -11.68],
                        5: [17.08, -17.66], 6: [22.49, -23.95], 7: [28.68, -31.24],
                        8: [32.27, -35.46], 16: [34.26, -37.60], 32: [40.60, -45.33]}
    return coefficient_dict[bits]


class SAWBTensor(nn.Module):
    """docstring for QuantMeasure."""

    def __init__(self, bits=8, aw='', name=''):
        super(SAWBTensor, self).__init__()

        self.aw = aw
        self.bits = bits
        self.c1, self.c2 = get_sawb_coefficients(self.bits)
        self.name = name

    def forward(self, input):
        if input.max() > -1e-8 and input.min() < 1e-8:
            Qn = -2 ** (self.bits - 1)
            Qp = 2 ** (self.bits - 1)
        elif input.max() > -1e-8 and input.min() > -1e-8:
            Qn = 0
            Qp = 2 ** self.bits - 1
        else:
            print("min max not compatible for SAWB")
            Qn = 0
            Qp = 0

        return UniformQuantizeSawb().apply(input, self.c1, self.c2, Qp, Qn)


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
        self.quantize_input = QuantMeasure()
        self.name = name

        self.first_pass = False
        self._build_weight_clip_val(qconfig.weight_quant_method, init_val=qconfig.clip_value)
        self._build_input_clip_val(qconfig.input_quant_method, init_val=qconfig.clip_value)

        # if self.name != "feedForward":
        if qconfig.householder:
            self.HM = HouseholderMultiplier(group=qconfig.hadamard_group, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
            self.HMinv = HouseholderInverseMultiplier(group=qconfig.hadamard_group, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
        if qconfig.householder:
            self.HM_w = HouseholderMultiplier(group=qconfig.hadamard_group, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
            self.HMinv_w = HouseholderInverseMultiplier(group=qconfig.hadamard_group, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
        # elif self.name == "feedForward":
        #     if qconfig.householder:
        #         self.HM = HouseholderMultiplier(group=256, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
        #         self.HMinv = HouseholderInverseMultiplier(group=256, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
        #     if qconfig.householder:
        #         self.HM_w = HouseholderMultiplier(group=256, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)
        #         self.HMinv_w = HouseholderInverseMultiplier(group=256, dim=self.weight.shape[-1], learnable=qconfig.learnable_hadamard)


        self.is_second = False
        self.epsilon = None
        if qconfig.track_step_size:
            self.active_track, self.weight_track, self.iter_track = [], [], []

        self.saved_activation = None

        if self.name == "feedForward":
            print(self.weight.shape[-1])

    def _build_weight_clip_val(self, quant_method, init_val, example=None):
        if quant_method == 'uniform':
            # init_val = self.weight.mean().item() + 3 * self.weight.std().item()
            self.register_buffer('weight_clip_val', torch.tensor([-init_val, init_val]))
            self.weight_clip_val = nn.Parameter(self.weight_clip_val)
        elif quant_method == 'lsq':
            # TODO: for now we abuse the name for consistent reference in learner.
            # assert learnable, 'LSQ must use leranable step size!'
            if qconfig.lsq_layerwise_weight == 'layer':
                self.weight_clip_val = LsqStepSize(
                    torch.tensor(1.0, requires_grad=True))  # stepsize will be initialized in the first quantization
            elif qconfig.lsq_layerwise_weight == 'row':

                self.weight_clip_val = LsqStepSize(
                    torch.ones_like(self.weight[:, 0]).clone().detach().requires_grad_(
                        True))  # stepsize will be initialized in the first quantization
            elif qconfig.lsq_layerwise_weight == 'column':
                if example is not None:
                    self.weight_clip_val = LsqStepSize(
                        torch.ones_like(example[0, :]).clone().detach().requires_grad_(
                            True))  # stepsize will be initialized in the first quantization
                else:
                    self.weight_clip_val = LsqStepSize(
                        torch.ones_like(self.weight[0, :]).clone().detach().requires_grad_(
                            True))  # stepsize will be initialized in the first quantization

        else:
            self.register_buffer('weight_clip_val', None)

    def _build_input_clip_val(self, quant_method, init_val, example=None):
        # print(quant_method, init_val)
        if quant_method == 'uniform':
            self.register_buffer('input_clip_val', torch.tensor([-init_val, init_val]))
            self.input_clip_val = nn.Parameter(self.input_clip_val)
        elif quant_method == 'lsq':
            # TODO: for now we abuse the name for consistent reference in learner.
            # assert learnable, 'LSQ must use learnable step size!'
            if qconfig.lsq_layerwise_input == 'layer':
                self.input_clip_val = LsqStepSize(
                    torch.tensor(1.0, requires_grad=True))  # stepsize will be initialized in the first quantization

            elif qconfig.lsq_layerwise_input == 'column':
                if example is not None:
                    if len(example.shape) == 3:
                        self.input_clip_val = LsqStepSize(
                            torch.ones_like(example[0, 0, :]).clone().detach().requires_grad_(
                                True))  # stepsize will be initialized in the first quantization
                    if len(example.shape) == 2:
                        self.input_clip_val = LsqStepSize(
                            torch.ones_like(example[0, :]).clone().detach().requires_grad_(
                                True))  # stepsize will be initialized in the first quantization
                else:
                    self.input_clip_val = LsqStepSize(
                        torch.ones_like(self.weight[0, :]).clone().detach().requires_grad_(
                            True))  # stepsize will be initialized in the first quantization

        else:
            self.register_buffer('input_clip_val', None)

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
            if qconfig.input_quant_method == 'lsq':
                self.input_clip_val.change_abs()
            if qconfig.weight_quant_method == 'lsq':
                self.weight_clip_val.change_abs()

        if qconfig.track_step_size and self.first_pass:
            self.active_track.append(self.input_clip_val.cpu().detach().numpy())
            self.weight_track.append(self.weight_clip_val.cpu().detach().numpy())
            self.iter_track.append(len(self.iter_track))

        if not self.first_pass:
            print("Actually Using QLinear!")
            self.first_pass = True

            if qconfig.householder:
                with torch.no_grad():
                    raw_input = deepcopy(input.detach())
                    best_hadamard_group, min_hadamard_quantize_error = 0, 1e10
                    for H_search in qconfig.group_search:
                        if input.shape[-1] % H_search != 0:
                            continue
                        H_search_input = raw_input
                        self.HM = HouseholderMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                        learnable=qconfig.learnable_hadamard)
                        self.HMinv = HouseholderInverseMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                                  learnable=qconfig.learnable_hadamard)
                        self.HM_w = HouseholderMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                        learnable=qconfig.learnable_hadamard)
                        self.HMinv_w = HouseholderInverseMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                                  learnable=qconfig.learnable_hadamard)
                        if qconfig.draw_value:
                            self.draw(H_search_input.detach(), f"{H_search}_input_search_hadamard", "1 origin")
                        if qconfig.householder:
                            H_search_input = self.HM(H_search_input)
                        if qconfig.draw_value:
                            self.draw(H_search_input.detach(), f"{H_search}_input_search_hadamard", "2 hadamard")
                        if qconfig.weight_quant_method == 'ptq':
                            qinput = quantize(H_search_input, qconfig.weight_preconditioner())
                        else:
                            self.input_clip_val.initialized = False
                            qinput = weight_quant_fn(H_search_input, self.input_clip_val, num_bits=qconfig.weight_num_bits,
                                                     symmetric=True,
                                                     quant_method=qconfig.weight_quant_method,
                                                     layerwise=qconfig.lsq_layerwise_weight,
                                                     learnable=qconfig.learnable_step_size)
                            self.input_clip_val.initialized = False
                        if qconfig.draw_value:
                            self.draw(qinput.detach(), f"{H_search}_input_search_hadamard", "3 quantize")
                        if qconfig.householder:
                            qinput = self.HMinv(qinput)
                        if qconfig.draw_value:
                            self.draw(qinput.detach(), f"{H_search}_input_search_hadamard", "4 hadamard inv")


                        if qconfig.draw_value:
                            self.draw(self.weight.detach(), f"{H_search}_weight", "1 origin")

                        if qconfig.householder_weight:
                            weight_in = self.HM_w(self.weight)

                            if qconfig.draw_value:
                                self.draw(weight_in.detach(), f"{H_search}_weight", "2 hadamard")

                        if qconfig.weight_quant_method == 'ptq':
                            qweight = quantize(self.weight, qconfig.weight_preconditioner())
                        else:
                            self.weight_clip_val.initialized = False
                            qweight = weight_quant_fn(weight_in, self.weight_clip_val, num_bits=qconfig.weight_num_bits,
                                                      symmetric=True,
                                                      quant_method=qconfig.weight_quant_method,
                                                      layerwise=qconfig.lsq_layerwise_weight,
                                                      learnable=qconfig.learnable_step_size)
                            self.weight_clip_val.initialized = False

                        if qconfig.draw_value:
                            self.draw(qweight.detach(), f"{H_search}_weight", "3 quantize")

                        if qconfig.householder_weight:
                            qweight = self.HMinv_w(qweight)

                        if qconfig.draw_value:
                            self.draw(qweight.detach(), f"{H_search}_weight", "4 hadamard")

                        qoutput = F.linear(qinput, qweight, self.bias)
                        output = F.linear(raw_input, self.weight, self.bias)
                        Q_error_o = (qoutput - output).norm()

                        Q_error_a, Q_error_w = (qinput - raw_input).norm(), (qweight - self.weight).norm()
                        Q_error = Q_error_a * Q_error_w * Q_error_o
                        if Q_error < min_hadamard_quantize_error:
                            min_hadamard_quantize_error = Q_error
                            best_hadamard_group = H_search



                        print(f"when group is {H_search} the error is A: {Q_error_a} and W: {Q_error_w} and O: {Q_error_o}")

                    self.HM = HouseholderMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                    learnable=qconfig.learnable_hadamard)
                    self.HMinv = HouseholderInverseMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                              learnable=qconfig.learnable_hadamard)
                    self.HM_w = HouseholderMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                    learnable=qconfig.learnable_hadamard)
                    self.HMinv_w = HouseholderInverseMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                              learnable=qconfig.learnable_hadamard)

                    print(f"At layer   {self.name}   The best group is {best_hadamard_group}")

        if qconfig.draw_value:
            self.draw(input.detach(), "input", "1 origin")

        if qconfig.householder:
            input = self.HM(input)

        if qconfig.draw_value:
            self.draw(input.detach(), "input", "2 hadamard")

        if qconfig.input_quant_method == 'ptq' or (self.name == 'addNorm_nsy' and qconfig.retain_large_value):
            qinput = self.quantize_input(input)
            # print(qinput.min())
        else:
            qinput = act_quant_fn(input, self.input_clip_val, num_bits=qconfig.activation_num_bits,
                                  symmetric=self.name != 'addNorm_nsy' or qconfig.householder,
                                  quant_method=qconfig.input_quant_method, layerwise=qconfig.lsq_layerwise_input,
                                  learnable=qconfig.learnable_step_size)

        if qconfig.draw_value:
            self.draw(qinput.detach(), "input", "3 quantize")

        if qconfig.householder and not qconfig.real_quantize:
            qinput = self.HMinv(qinput)

        if qconfig.draw_value:
            self.draw(qinput.detach(), "input", "4 hadamard inv")

        qbias = self.bias

        if qconfig.draw_value:
            self.draw(self.weight.detach(), "weight", "1 origin")

        if qconfig.householder_weight:
            weight_in = self.HM_w(self.weight)

            if qconfig.draw_value:
                self.draw(weight_in.detach(), "weight", "2 hadamard")
        else:
            weight_in = self.weight

        if qconfig.weight_quant_method == 'ptq':
            qweight = quantize(self.weight, qconfig.weight_preconditioner())
        else:
            qweight = weight_quant_fn(weight_in, self.weight_clip_val, num_bits=qconfig.weight_num_bits,
                                      symmetric=True,
                                      quant_method=qconfig.weight_quant_method, layerwise=qconfig.lsq_layerwise_weight,
                                      learnable=qconfig.learnable_step_size)

        if qconfig.draw_value:
            self.draw(qweight.detach(), "weight", "3 quantize")

        if qconfig.householder_weight and not qconfig.real_quantize:
            qweight = self.HMinv_w(qweight)

        if qconfig.draw_value:
            self.draw(qweight.detach(), "weight", "4 hadamard")

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = F.linear(qinput, qweight, qbias)
        else:
            output = linear_act.apply(qinput, qweight, qbias)

        if qconfig.draw_value:
            self.draw(output.detach(), "output", "1 quantize")

        if qconfig.householder:
            assert self.HM.H.shape == self.HMinv.H_inv.shape, print(self.HM.H.shape, self.HMinv.H_inv.shape)
            assert self.HM_w.H.shape == self.HMinv_w.H_inv.shape, print(self.HM_w.H.shape, self.HMinv_w.H_inv.shape)

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

            if qconfig.householder:
                raw_input = deepcopy(input.detach())
                best_hadamard_group, min_hadamard_quantize_error = 0, 1e10
                for H_search in qconfig.group_search:
                    if input.shape[-1] % H_search != 0:
                        continue
                    H_search_input = raw_input
                    self.HM = HouseholderMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                    learnable=qconfig.learnable_hadamard)
                    self.HMinv = HouseholderInverseMultiplier(group=H_search, dim=H_search_input.shape[-1],
                                                              learnable=qconfig.learnable_hadamard)
                    if qconfig.draw_value:
                        self.draw(H_search_input.detach(), f"{H_search}_input_search_hadamard", "1 origin")
                    if qconfig.householder:
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
                    if qconfig.householder:
                        qinput = self.HMinv(qinput)
                    if qconfig.draw_value:
                        self.draw(qinput.detach(), f"{H_search}_input_search_hadamard", "4 hadamard inv")

                    Q_error = (qinput - raw_input).norm()
                    if Q_error < min_hadamard_quantize_error:
                        min_hadamard_quantize_error = Q_error
                        best_hadamard_group = H_search

                    # print(f"when group is {H_search} the error is {Q_error}")

                self.HM = HouseholderMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                learnable=qconfig.learnable_hadamard)
                self.HMinv = HouseholderInverseMultiplier(group=best_hadamard_group, dim=H_search_input.shape[-1],
                                                          learnable=qconfig.learnable_hadamard)

                print(f"At layer   {self.name}   The best group is {best_hadamard_group}")

        if qconfig.draw_value:
            self.draw(input.detach(), "input", "1 origin")

        if qconfig.householder:
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

        if qconfig.householder:
            qinput = self.HMinv(qinput)

        if qconfig.draw_value:
            self.draw(qinput.detach(), "input", "4 hadamard inv")

        if hasattr(self, 'exact') or not qconfig.quantize_gradient:
            output = qinput
        else:
            output = identity_act.apply(qinput)

        # exit(0)

        return output
