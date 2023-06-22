import torch
import math
import time
import numpy as np
from tqdm import trange
import torch.nn as nn
import IPython

def checkNAN(x, s=''):
    N = torch.isnan(x)
    cN = torch.count_nonzero(N)
    if cN != 0:
        print("NAN!!!{}".format(s))
        print(cN)
        print(x.shape)
        print(x)


def hadamard(src, tar):
    N = src.shape[0]
    v = tar - src
    v = v / v.norm()
    return torch.eye(N) - 2 * v.view(N, 1) @ v.view(1, N)




class Preconditioner:
    def __init__(self, x, num_bits, left=True):
        self.left = left
        self.x_shape = x.shape
        self.num_bins = 2 ** num_bits - 1
        self.num_bits = num_bits

        self.Qn = 0
        self.Qp = self.num_bins

        self.x = self.flatten(x)
        self.Tx = self.transform(self.x)

    def flatten(self, x):
        self.x_shape2 = x.shape
        self.x_shape_double = torch.cat([x, x], dim=0).shape
        return x.reshape(x.shape[0], -1)

    def deflatten(self, Tx):
        try:
            x = Tx.view(*self.x_shape2)
        except:
            x = Tx.view(*self.x_shape_double)
        return x

    def forward(self):
        return self.Tx

    def inverse(self, Tx):
        x = self.inverse_transform(Tx)
        return self.deflatten(x)


class ScalarPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        qzero = -self.zero_point * self.scale
        iqzero = torch.floor(qzero)
        if iqzero > 0:
            mx = (iqzero - self.num_bins) * mn / iqzero
        elif iqzero == 0:
            self.zero_point1, mn = 0, 0
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point


class ScalarPreconditionerAct(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(ScalarPreconditionerAct, self).__init__(x, num_bits, left)

    def transform(self, x):
        with torch.no_grad():
            mn = x.min() - 1e-8
            mx = x.max() + 1e-8

        self.zero_point = mn
        self.scale = self.num_bins / (mx - mn)

        return (x - self.zero_point) * self.scale

    def inverse_transform(self, x):
        return x / self.scale + self.zero_point

class SingleDivideGWPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(SingleDivideGWPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x, debug=False):
        torch.set_printoptions(linewidth=100)

        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

            # mn = torch.minimum(x.min(dim=0)[0] - 1e-8, torch.zeros_like(x.min(dim=0)[0]))
            # mx = torch.maximum(x.max(dim=0)[0] + 1e-8, torch.zeros_like(x.max(dim=0)[0]))

        self.num_bins_half = 2 ** self.num_bits - 2

        # print(mn.abs(), mx.abs())
        self.scale1 = self.num_bins_half / (2 * max(mn.abs(), mx.abs()))
        # self.scale1 = self.num_bins_half / (2 * torch.maximum(mn.abs(), mx.abs())).unsqueeze(0)
        # print("GW", self.scale1.squeeze(-1).sort()[0], self.scale1.shape)

        if torch.isnan(self.scale1.max()):
            print("save for 1 nan not save yet")
            exit(0)

        self.Qn = -torch.ones_like(x) * (self.num_bins_half // 2)
        self.Qp = torch.ones_like(x) * (self.num_bins_half // 2)

        output = x * self.scale1

        return output


    def inverse_transform(self, x):

        dequantize = x / self.scale1

        return dequantize

class SingleDivideGAPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(SingleDivideGAPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x, debug=False):
        torch.set_printoptions(linewidth=100)

        with torch.no_grad():

            mn = torch.minimum(x.min(dim=1)[0] - 1e-8, torch.zeros_like(x.min(dim=1)[0]))
            mx = torch.maximum(x.max(dim=1)[0] + 1e-8, torch.zeros_like(x.max(dim=1)[0]))

        self.num_bins_half = 2 ** self.num_bits - 2
        self.scale1 = self.num_bins_half / (2 * torch.maximum(mn.abs(), mx.abs())).unsqueeze(-1)

        # print("GW", self.scale1.squeeze(-1).sort()[0], self.scale1.shape)
        if torch.isnan(self.scale1.max()):
            print("save for 1 nan not save yet")
            exit(0)

        self.Qn = -torch.ones_like(x) * (self.num_bins_half // 2)
        self.Qp = torch.ones_like(x) * (self.num_bins_half // 2)

        output = x * self.scale1
        return output


    def inverse_transform(self, x):
        dequantize = x / self.scale1
        return dequantize

class TwoLayerWeightPreconditioner(Preconditioner):
    # Y = D (Y - z 1^\top)
    # X = D^-1 Y + z 1^\top
    def __init__(self, x, num_bits, left=True):
        super(TwoLayerWeightPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x, debug=False):
        with torch.no_grad():
            mn = min(x.min() - 1e-8, 0)
            mx = max(x.max() + 1e-8, 0)

        self.num_bins = 2 ** (2 * self.num_bits) - 1
        self.num_bins_half = 2 ** self.num_bits - 2
        # self.num_bins_half = 2 ** self.num_bits - 2
        # self.num_bins = self.num_bins_half * (self.num_bins_half)

        self.scale1 = self.num_bins / (2 * max(mn.abs(), mx.abs()))

        if torch.isnan(self.scale1):
            print("save for 1 nan not save yet")

            exit(0)

        # self.Qn1, self.Qn2 = -torch.ones_like(x) * (self.num_bins_half - 1) / 2, -torch.ones_like(x) * (self.num_bins_half - 1) / 2
        # self.Qp1, self.Qp2 = torch.ones_like(x) * (self.num_bins_half - 1) / 2, torch.ones_like(x) * (self.num_bins_half - 1) / 2

        self.Qn1, self.Qn2 = -torch.ones_like(x) * (self.num_bins_half // 2), -torch.ones_like(x) * (self.num_bins_half // 2)
        self.Qp1, self.Qp2 = torch.ones_like(x) * (self.num_bins_half // 2), torch.ones_like(x) * (self.num_bins_half // 2)

        self.Qn = torch.cat([self.Qn1, self.Qn2], dim=0)
        self.Qp = torch.cat([self.Qp1, self.Qp2], dim=0)

        first_transform_0 = x * self.scale1
        first_transform = torch.round(torch.clamp(first_transform_0 / self.num_bins_half, self.Qn1, self.Qp1))
        second_transform = first_transform_0 - first_transform * self.num_bins_half
        output = torch.cat([first_transform, second_transform], dim=0)

        return output

    def inverse_transform(self, x):
        half_shape = int(x.shape[0] / 2)
        first, second = torch.split(x, [half_shape, half_shape], dim=0)
        first = first * self.num_bins_half / self.scale1
        second = second / self.scale1
        dequantize = torch.cat([first, second], dim=0)

        return dequantize

class LUQPreconditioner(Preconditioner):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, x, num_bits, left=True):
        super(LUQPreconditioner, self).__init__(x, num_bits, left)

    def transform(self, x):
        # print(x.max(), x.min(), x.argmax(keepdim=False), x.argmin(keepdim=False))
        # print(x[x.argmax() // x.shape[1] - 8: x.argmax() // x.shape[1] + 2,
        #         x.argmax() % x.shape[1] - 5:x.argmax() % x.shape[1] + 5])
        # print(x.argmax() // x.shape[1], x.argmax() % x.shape[1], x.shape[0], x.shape[1])
        # print('_'*20)
        self.debug = False

        with torch.no_grad():

            mx = x.abs().max()
            self.max_bins = 2 ** (self.num_bits - 1)
            alpha = mx / 2 ** self.max_bins

            self.minivalue = 2 ** (-self.max_bins - 3)
            self.num_bins = self.max_bins + 1
            if self.debug:
                print(mx, alpha)
        if self.debug:
            print(x)
        sign = (x > 0)
        sign11 = sign.int() * 2 - torch.ones_like(x)
        if self.debug:
            print("sign", sign)
        thres = (x.abs() > alpha)
        sample_prob = (~thres) * x.abs() / alpha
        checkNAN(sample_prob, "sample prob")
        prob = torch.bernoulli(sample_prob)
        if self.debug:
            print("prob", prob)
        T = x * thres + sign11 * alpha * prob
        self.mid_ckpt = T

        self.alpha = alpha
        #
        if self.debug:
            print("T", T)

        output = T / alpha

        if self.debug:
            print("output", output)

        checkNAN(output, "luq output before log")
        logx, nearzero = self.log_with_0(output.abs())

        self.sign11 = sign11
        self.nearzero = nearzero
        output = logx * ~nearzero

        if self.debug:
            print("output log", output)
        checkNAN(output, "luq output after log")
        return output

    def log_with_0(self, x):
        small = (x < self.minivalue)
        small11 = small.int() * 2 - torch.ones_like(x)
        if self.debug:
            print("small", small)
        x = x + small * self.minivalue
        if self.debug:
            print("small x", x)
        logx = torch.log2(x) + torch.ones_like(x)
        if self.debug:
            print("logx", logx)
        return logx, small

    def exp_with_0(self, x, nearzero):
        powx = torch.pow(2, x) / 2
        if self.debug:
            print("pow x", powx)
        # print("nearzer0", nearzero)
        x = powx * ~nearzero
        if self.debug:
            print("pow nearzero x", x)
        return x

    def inverse_transform(self, x):
        if self.debug:
            print("inverse x", x)

        x = self.exp_with_0(x.abs(), self.nearzero) * self.sign11

        if self.debug:
            print("x", x)
        checkNAN(x, "luq inverse x")
        if self.debug:
            print(x)
        output = x * self.alpha

        if self.debug:
            print("final", output)

        return output


class HadamardMultiplier(nn.Module):
    # y = (x - z) * scale
    # x = y / scale + z
    def __init__(self, group, dim, learnable):
        super(HadamardMultiplier, self).__init__()
        self.group = group
        self.dim = dim
        H_group = self.constructH(group)
        self.H = nn.Parameter(H_group.repeat(dim // group, 1, 1), requires_grad=learnable)

    def constructH(self, group):
        H = torch.ones(1, 1).cuda()

        for i in range(int(math.log2(group))):
            H = torch.cat((torch.cat([H, H], 1),
                           torch.cat([H, -H], 1)), 0) / math.sqrt(2)
        assert H.shape[0] == group
        return H

    def forward(self, x):
        x_shape2 = x.shape
        x = x.reshape(-1, x.shape[-1])

        x = x.reshape(-1, self.dim // self.group, self.group).transpose(0, 1)
        x = torch.bmm(x, self.H).transpose(0, 1)
        # H = torch.block_diag(*self.H)
        # x = torch.mm(x, H)

        x = x.reshape(x_shape2)
        return x

