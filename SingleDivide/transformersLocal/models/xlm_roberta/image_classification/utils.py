import os
import numpy as np
import torch
import shutil
import torch.distributed as dist
import matplotlib.pyplot as plt
import time

def should_backup_checkpoint(args):
    def _sbc(epoch):
        return args.gather_checkpoints # and (epoch < 10 or epoch % 10 == 0)
    return _sbc

def checkNAN(x, s=''):
    N = torch.isnan(x)
    cN = torch.count_nonzero(N)
    if cN != 0:

        print("NAN!!!{}".format(s))
        print(cN)
        print(x.shape)
        print(x[:, 0])
        return True
    return False


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar', checkpoint_dir='./', backup_filename=None):
    if (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0:
        filename = os.path.join(checkpoint_dir, filename)
        print("SAVING {}".format(filename))
        torch.save(state, filename)
        if is_best:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, 'model_best.pth.tar'))
        if backup_filename is not None:
            shutil.copyfile(filename, os.path.join(checkpoint_dir, backup_filename))



def timed_generator(gen):
    start = time.time()
    for g in gen:
        end = time.time()
        t = end - start
        yield g, t
        start = time.time()


def timed_function(f):
    def _timed_function(*args, **kwargs):
        start = time.time()
        ret = f(*args, **kwargs)
        return ret, time.time() - start
    return _timed_function


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
    return rt


def dict_add(x, y):
    if x is None:
        return y
    return {k: x[k] + y[k] for k in x}


def dict_minus(x, y):
    return {k: x[k] - y[k] for k in x}


def dict_sqr(x):
    return {k: x[k]**2 for k in x}


def dict_sqrt(x):
    return {k: torch.sqrt(x[k]) for k in x}


def dict_mul(x, a):
    return {k: x[k]*a for k in x}


def dict_clone(x):
    return {k: x[k].clone() for k in x}


def twolayer_linearsample_weight(m1, m2):
    m2 = torch.cat([m2, m2], dim=0)
    m1_len = torch.linalg.norm(m1, dim=1)
    m2_len = torch.linalg.norm(m2, dim=1)
    vec_norm = m1_len.mul(m2_len)

    # if vec_norm.shape[0] < 200:
    #     print(" too small for {}!".format(vec_norm.shape[0]))
    #
    # if torch.rand(1) < 0.0003:
    #     print("weight {}".format(vec_norm.shape[0]))


    index, norm_x = sample_index_from_bernouli(vec_norm)
    norm_x[norm_x == 0] = 1e-10
    m1 = m1 / norm_x.unsqueeze(1)

    m1, m2 = m1[index, :], m2[index, :]

    # if torch.rand(1) < 0.002:
    #     draw(norm_x, 'plt/weight/')

    return m1, m2



def sample_index_from_bernouli(x):
    # if torch.isnan(x[0]):
    #     print("aduiduidui")
    #     exit(0)
    # print(x.max(), x.min(), x.mean())
    len_x = len(x)
    norm_x = x * len_x / (2 * x.sum())
    # print(norm_x)
    typeflag = 'NoNoNo'
    randflag = torch.rand(1)

    cnt = 0
    while norm_x.max() > 1 and cnt < len_x / 2:
        small_index = torch.nonzero((norm_x < 1)).squeeze()
        small_value = norm_x[small_index]
        cnt = len_x - len(small_index)
        norm_x = torch.clamp(norm_x, 0, 1)
        if small_value.max() == 0 and small_value.min() == 0:
            break
        # print(len(x), cnt)
        small_value = small_value * (len_x // 2 - cnt) / small_value.sum()
        norm_x[small_index] = small_value

        # print("small index is {}, \n small value is {}, cnt is {}".format(small_index, small_value, cnt))
        # print("norm x is {}".format(norm_x))
        # print("sorted norm x is {}".format(norm_x.sort()[0]))
        # print("small index is {}, \n small value is {}, cnt is {}".format(small_index, small_value, cnt))
        # print("sum up to {}".format(norm_x.sum()))
        # print("cnt is {}".format(cnt))
        # print("_______________________________________________________________________________________________________")
        # exit(0)
    # if norm_x.max() > 1 or norm_x.min() < 0:
    #     # if torch.isnan(norm_x[0]):
    #     typeflag = 'debug'
    #     print("We change it to debug mode because of the Bernoulli")
    # if typeflag == 'debug':
    #     with open("debug.txt", "a") as f:
    #         f.write("raw {} is {}\n".format(randflag, x))
    #     with open("debug.txt", "a") as f:
    #         f.write("the after norm {} is {}\n".format(randflag, norm_x))
    # # print("norm x is {}".format(norm_x))

    checkNAN(norm_x, 'norm_x')
    sample_index = torch.bernoulli(norm_x)
    # print("sample_index is {}
    # if typeflag == 'debug':
    #     with open("debug.txt", "a") as f:
    #         f.write("sample index {} is {}\n".format(randflag, sample_index))
    # # index = [x for x in range(len(sample_index)) if sample_index[x] == 1]
    # # try:
    # if sample_index.max() > 1 or sample_index.min() < 0:
    #     print(sample_index)
    #     print(x)

    index = torch.nonzero((sample_index == 1)).squeeze()
    # if typeflag == 'debug':
    #     with open("debug.txt", "a") as f:
    #         f.write("index {} is {}\n".format(randflag, index))
    # print("bernoulli", x, '\n', index, '\n', norm_x, '\n', len(index))
    return index, norm_x

def twolayer_linearsample_input(m1, m2):
    m1clone, m2clone = m1.clone(), m2.clone()

    m2 = torch.cat([m2, m2], dim=0)
    m1_len = torch.linalg.norm(m1, dim=1)
    m2_len = torch.linalg.norm(m2, dim=1)
    vec_norm = m1_len.mul(m2_len)
    # print("vecnorm", vec_norm)

    # if torch.rand(1) < 0.0003:
    #     print("input {}".format(vec_norm.shape[0]))

    index, norm_x = sample_index_from_bernouli(vec_norm)
    # print("index", index, '\n', "normx", norm_x)
    norm_x[norm_x == 0] = 1e-10
    m1 = m1 / norm_x.unsqueeze(1)

    Ind = torch.zeros_like(m1)
    Ind[index] = 1
    m1 = m1.mul(Ind)
    m1 = m1[0:m1.shape[0] // 2] + m1[m1.shape[0] // 2:]

    # m1, m2 = m1[index, :], m2[index, :]
    if checkNAN(m1):
        print("save for m1m2")

    # if torch.rand(1) < 0.005:
    #     draw(norm_x, 'plt/input/')
    return m1, m2

def draw(x, s=''):
    x = x.sort(descending=True)[0]
    x = x.detach().cpu().numpy()

    plt_x = np.arange(len(x))
    plt_y = np.array([])
    for i in range(len(x)):
        plt_y = np.append(plt_y, np.sum(x[:i]) / np.sum(x))

    plt.figure()
    plt.plot(plt_x, plt_y)
    plt.axvline(x=len(x) // 2, color='red', linestyle='--')
    time_tuple = time.localtime(time.time())
    plt.savefig('{}{}:{}:{}.png'.format(s, time_tuple[3], time_tuple[4], time_tuple[5]))

    plt.figure()
    plt.plot(plt_x, np.log2(plt_y))
    plt.axvline(x=len(x) // 2, color='red', linestyle='--')
    plt.savefig('{}{}:{}:{}_log.png'.format(s, time_tuple[3], time_tuple[4], time_tuple[5]))

    print("savefig")
