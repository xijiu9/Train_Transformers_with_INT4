import torch
from .quantize import qconfig
from .utils import *
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import pickle
from matplotlib.colors import LogNorm


def get_error_grad(m):
    grad_dict = {}

    if hasattr(m, 'layer4'):
        layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        layers = [m.layer1, m.layer2, m.layer3]

    for lid, layer in enumerate(layers):
        for bid, block in enumerate(layer):
            clayers = [block.conv1_in, block.conv2_in]
            if hasattr(block, 'conv3'):
                clayers.extend([block.conv3_in])

            for cid, clayer in enumerate(clayers):
                layer_name = 'conv_{}_{}_{}_error'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.grad.detach().cpu()

    return grad_dict


def get_grad(m):
    grad_dict = {}

    if hasattr(m, 'layer4'):
        layers = [m.layer1, m.layer2, m.layer3, m.layer4]
    else:
        layers = [m.layer1, m.layer2, m.layer3]

    for lid, layer in enumerate(layers):
        for bid, block in enumerate(layer):
            clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                else [block.conv1, block.conv2]

            for cid, clayer in enumerate(clayers):
                layer_name = 'conv_{}_{}_{}_weight'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.detach().cpu()
                layer_name = 'conv_{}_{}_{}_grad'.format(lid + 1, bid + 1, cid + 1)
                grad_dict[layer_name] = clayer.weight.grad.detach().cpu()

    return grad_dict


def get_batch_grad(model_and_loss, optimizer, val_loader, ckpt_name):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    optimizer.zero_grad()
    cnt = 0
    for i, (input, target) in data_iter:
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        cnt += 1

    for param_group in optimizer.param_groups:
        for param in param_group['params']:
            param.grad /= cnt

    grad = get_grad(m)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        torch.save(grad, ckpt_name)
    return get_grad(m)


def get_grad_bias_std(model_and_loss, optimizer, val_loader, mean_grad, ckpt_name, num_epochs=1):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    empirical_mean_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss, output = model_and_loss(input, target)
            loss.backward()
            torch.cuda.synchronize()

            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                grad_dict = get_grad(m)

                e_grad = dict_sqr(dict_minus(grad_dict, mean_grad))
                if var_grad is None:
                    var_grad = e_grad
                else:
                    var_grad = dict_add(var_grad, e_grad)

                if empirical_mean_grad is None:
                    empirical_mean_grad = grad_dict
                else:
                    empirical_mean_grad = dict_add(empirical_mean_grad, grad_dict)

            cnt += 1

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        std_grad = dict_sqrt(dict_mul(var_grad, 1.0 / cnt))
        bias_grad = dict_minus(dict_mul(empirical_mean_grad, 1.0/cnt), mean_grad)
        torch.save(std_grad, ckpt_name)
        return bias_grad, std_grad


def get_grad_std_naive(model_and_loss, optimizer, val_loader, num_epochs=1):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        qconfig.quantize_gradient = False
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        mean_grad = dict_clone(get_grad(m))

        qconfig.quantize_gradient = True
        for e in range(num_epochs):
            optimizer.zero_grad()
            loss, output = model_and_loss(input, target)
            loss.backward()
            torch.cuda.synchronize()
            grad_dict = get_grad(m)

            e_grad = dict_sqr(dict_minus(grad_dict, mean_grad))
            if var_grad is None:
                var_grad = e_grad
            else:
                var_grad = dict_add(var_grad, e_grad)

            cnt += 1

    std_grad = dict_sqrt(dict_mul(var_grad, 1.0 / cnt))
    return std_grad


def debug_bias(model_and_loss, optimizer, val_loader):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    data_iter = enumerate(val_loader)
    var_grad = None
    empirical_mean_grad = None
    cnt = 0
    for i, (input, target) in data_iter:
        break

    qconfig.quantize_gradient = False
    optimizer.zero_grad()
    loss, output = model_and_loss(input, target)
    loss.backward()
    torch.cuda.synchronize()

    exact_grad = get_grad(m)
    empirical_mean_grad = None
    qconfig.quantize_gradient = True
    for e in range(100):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()

        cnt += 1
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            grad_dict = get_grad(m)

            if empirical_mean_grad is None:
                empirical_mean_grad = grad_dict
            else:
                empirical_mean_grad = dict_add(empirical_mean_grad, grad_dict)

            bias_grad = dict_minus(dict_mul(empirical_mean_grad, 1.0/cnt), exact_grad)
            print(e, bias_grad['conv_1_1_1_grad'].abs().mean())


def get_gradient(model_and_loss, optimizer, input, target, prefix):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        grad_dict = get_grad(m)
        ckpt_name = "{}_weight.grad".format(prefix)
        torch.save(grad_dict, ckpt_name)

    grad_dict = get_error_grad(m)
    if not torch.distributed.is_initialized():
        rank = 0
    else:
        rank = torch.distributed.get_rank()

    ckpt_name = "{}_{}_error.grad".format(prefix, rank)
    torch.save(grad_dict, ckpt_name)


def dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    qconfig.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    # print("Computing gradient std...")
    # get_grad_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad")

    print("Computing quantization noise...")
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/exact")

    qconfig.quantize_gradient = True
    for i in range(10):
        print(i)
        get_gradient(model_and_loss, optimizer, input, target, checkpoint_dir + "/sample_{}".format(i))

    # print("Computing quantized gradient std...")
    # get_grad_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad")


def key(a):
    return [int(i) for i in a.split('_')[1:4]]


def fast_dump(model_and_loss, optimizer, val_loader, checkpoint_dir):
    # debug_bias(model_and_loss, optimizer, val_loader)
    # exit(0)

    qconfig.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    print("Computing gradient std...")
    g_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad", num_epochs=1)

    qconfig.quantize_gradient = True
    q_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad", num_epochs=1)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        bias_grad, std_grad = g_outputs
        bias_quan, std_quan = q_outputs
        weight_names = list(grad.keys())
        weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
        weight_names = list(set(weight_names))
        weight_names.sort(key=key)
        for k in weight_names:
            grad_mean = grad[k + '_grad']
            sg = std_grad[k + '_grad']
            bg = bias_grad[k + '_grad']
            sq = std_quan[k + '_grad']
            bq = bias_quan[k + '_grad']

            print('{}, batch grad mean={}, sample std={}, sample bias={}, overall std={}, overall bias={}'.format(
                k, grad_mean.abs().mean(), sg.mean(), bg.abs().mean(), sq.mean(), bq.abs().mean()))


def fast_dump_2(model_and_loss, optimizer, val_loader, checkpoint_dir):
    qconfig.quantize_gradient = False
    print("Computing batch gradient...")
    grad = get_batch_grad(model_and_loss, optimizer, val_loader, checkpoint_dir + "/grad_mean.grad")

    print("Computing gradient std...")
    g_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std.grad", num_epochs=1)

    # config.quantize_gradient = True
    # q_outputs = get_grad_bias_std(model_and_loss, optimizer, val_loader, grad, checkpoint_dir + "/grad_std_quan.grad", num_epochs=3)
    std_quan = get_grad_std_naive(model_and_loss, optimizer, val_loader, num_epochs=10)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        bias_grad, std_grad = g_outputs
        # bias_quan, std_quan = q_outputs
        weight_names = list(grad.keys())
        weight_names = [n.replace('_grad', '').replace('_weight', '') for n in weight_names]
        weight_names = list(set(weight_names))
        weight_names.sort(key=key)

        sample_var = 0.0
        quant_var = 0.0
        for k in weight_names:
            grad_mean = grad[k + '_grad']
            sg = std_grad[k + '_grad']
            sq = std_quan[k + '_grad']

            print('{}, batch grad norm={}, sample var={}, quantization var={}, overall var={}'.format(
                k, grad_mean.norm()**2, sg.norm()**2, sq.norm()**2, sq.norm()**2 + sg.norm()**2))

            sample_var += sg.norm()**2
            quant_var += sq.norm()**2

        print('SampleVar = {}, QuantVar = {}, OverallVar = {}'.format(
            sample_var, quant_var, sample_var + quant_var))



def plot_bin_hist(model_and_loss, optimizer, val_loader):
    qconfig.grads = []
    qconfig.acts = []
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    loss, output = model_and_loss(input, target)

    optimizer.zero_grad()
    loss.backward()
    torch.cuda.synchronize()

    # fig, ax = plt.subplots(figsize=(5, 5))
    g = qconfig.grads[20]
    # ax.hist(g.cpu().numpy().ravel(), bins=2**config.backward_num_bits-1)
    # ax.set_yscale('log')
    # fig.savefig('grad_output_hist.pdf')

    for i in [1, 2]:
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.hist(g[i].cpu().numpy().ravel(), bins=256, range=[-1e-5, 1e-5])
        ax.set_yscale('log')
        ax.set_xlim([-1e-5, 1e-5])
        ax.set_xticks([-1e-5, 0, 1e-5])
        ax.set_xticklabels(['$-10^{-5}$', '$0$', '$10^{-5}$'])
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.05 * w, b, 0.95 * w, h])
        fig.savefig('{}_hist.pdf'.format(i), transparent=True)

    from image_classification.quantize import quantize

    def plot_each(preconditioner, Preconditioner, name, g):
        input = g
        prec = preconditioner(g, num_bits=qconfig.backward_num_bits)
        g = prec.forward()

        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.hist(g.cpu().numpy().ravel(), bins=2**qconfig.backward_num_bits-1, range=[0, 255])
        ax.set_yscale('log')
        ax.set_ylim([1, 1e6])
        ax.set_xlim([0, 255])
        ax.set_xticks([0, 255])
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.05 * w, b, 0.95 * w, h])
        fig.savefig('{}_hist.pdf'.format(name), transparent=True)

        prec.zero_point *= 0
        bin_sizes = []
        for i in range(128):
            bin_sizes.append(float(prec.inverse_transform(torch.eye(128)[:,i:i+1].cuda()).sum()))
        print(bin_sizes)
        fig, ax = plt.subplots(figsize=(2.5, 2.5))
        ax.hist(bin_sizes, bins=50, range=[0, 1e-5])
        # ax.set_yscale('log')
        ax.set_xlim([0, 1e-5])
        ax.set_xticks([0, 1e-5])
        ax.set_xticklabels(['$0$', '$10^{-5}$'])
        l, b, w, h = ax.get_position().bounds
        ax.set_position([l + 0.05 * w, b, 0.95 * w, h])
        # ax.set_ylim([0, 128])
        fig.savefig('{}_bin_size_hist.pdf'.format(name), transparent=True)

        gs = []
        for i in range(10):
            grad = quantize(input, Preconditioner, stochastic=True)
            gs.append(grad.cpu().numpy())
        var = np.stack(gs).var(0).sum()
        print(var)

    from image_classification.preconditioner import ScalarPreconditionerAct, DiagonalPreconditioner, BlockwiseHouseholderPreconditioner
    plot_each(ScalarPreconditionerAct, lambda x: ScalarPreconditionerAct(x, qconfig.backward_num_bits), 'PTQ', g)
    plot_each(DiagonalPreconditioner, lambda x: DiagonalPreconditioner(x, qconfig.backward_num_bits), 'PSQ', g)
    plot_each(BlockwiseHouseholderPreconditioner, lambda x: BlockwiseHouseholderPreconditioner(x, qconfig.backward_num_bits), 'BHQ', g)

    # R = g.max(1)[0] - g.min(1)[0]
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.hist(R.cpu().numpy().ravel(), bins=2 ** config.backward_num_bits - 1)
    # fig.savefig('dyn_range_hist.pdf')

    # prec = BlockwiseHouseholderPreconditioner(g, num_bits=config.backward_num_bits)
    # gH = prec.T @ g
    # R = gH.max(1)[0] - gH.min(1)[0]
    # fig, ax = plt.subplots(figsize=(5, 5))
    # ax.hist(R.cpu().numpy().ravel(), bins=2 ** config.backward_num_bits - 1)
    # fig.savefig('bH_dyn_range_hist.pdf')

    # if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
    #     num_grads = len(config.grads)
    #     fig, ax = plt.subplots(num_grads, figsize=(5, 5*num_grads))
    #     for i in range(num_grads):
    #         g = config.grads[i]
    #         ax[i].hist(g.cpu().numpy().ravel(), bins=2**config.backward_num_bits)
    #         ax[i].set_title(str(i))
    #         print(i, g.shape)
    #
    #     fig.savefig('grad_hist.pdf')

        # np.savez('errors.pkl', *config.grads)
        # np.savez('acts.pkl', *config.acts)


def plot_weight_hist(model_and_loss, optimizer, val_loader):
    qconfig.grads = []
    qconfig.acts = []
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:32]
    target = target[:32]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    m.set_debug(True)
    loss, output = model_and_loss(input, target)

    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        weights = []
        exact_weights = []
        acts = []
        names = []
        ins = []

        if hasattr(m, 'layer4'):
            layers = [m.layer1, m.layer2, m.layer3, m.layer4]
        else:
            layers = [m.layer1, m.layer2, m.layer3]

        print(m.layer1[0].conv1_out[0,10])
        print(m.layer1[0].conv1_bn_out[0,10])
        print(m.layer1[0].conv1_relu_out[0,10])
        print(m.layer1[0].conv2_in[0, 10])
        print(m.layer1[0].bn1.running_mean, m.layer1[0].bn1.running_var)

        for lid, layer in enumerate(layers):
            for bid, block in enumerate(layer):
                clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                    else [block.conv1, block.conv2]

                for cid, clayer in enumerate(clayers):
                    layer_name = 'conv_{}_{}_{}'.format(lid + 1, bid + 1, cid + 1)
                    names.append(layer_name)
                    exact_weights.append(clayer.weight.detach().cpu().numpy())
                    weights.append(clayer.qweight.detach().cpu().numpy())
                    acts.append(clayer.act.detach().cpu().numpy())
                    ins.append(clayer.iact.detach().cpu().numpy())

        num_weights = len(weights)
        fig, ax = plt.subplots(num_weights, figsize=(5, 5*num_weights))
        for i in range(num_weights):
            weight = weights[i]
            ax[i].hist(weight.ravel(), bins=2**qconfig.backward_num_bits)
            ax[i].set_title(names[i])
            print(i, weight.min(), weight.max())

        fig.savefig('weight_hist.pdf')
        np.savez('acts.pkl', *acts)
        np.savez('exact_weights.pkl', *exact_weights)
        np.savez('weights.pkl', *weights)
        np.savez('iacts.pkl', *ins)
        with open('layer_names.pkl', 'wb') as f:
            pickle.dump(names, f)

    qconfig.quantize_weights = False
    loss, output = model_and_loss(input, target)
    if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
        acts = []
        ins = []

        if hasattr(m, 'layer4'):
            layers = [m.layer1, m.layer2, m.layer3, m.layer4]
        else:
            layers = [m.layer1, m.layer2, m.layer3]

        for lid, layer in enumerate(layers):
            for bid, block in enumerate(layer):
                clayers = [block.conv1, block.conv2, block.conv3] if hasattr(block, 'conv3') \
                    else [block.conv1, block.conv2]

                for cid, clayer in enumerate(clayers):
                    layer_name = 'conv_{}_{}_{}'.format(lid + 1, bid + 1, cid + 1)
                    acts.append(clayer.act.detach().cpu().numpy())
                    ins.append(clayer.iact.detach().cpu().numpy())

        np.savez('exact_acts.pkl', *acts)
        np.savez('exact_iacts.pkl', *ins)


def write_errors(model_and_loss, optimizer, val_loader):
    data_iter = enumerate(val_loader)
    for i, (input, target) in data_iter:
        break

    input = input[:128]
    target = target[:128]

    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    for iter in range(10):
        print(iter)
        qonfig.grads = []
        loss, output = model_and_loss(input, target)
        optimizer.zero_grad()
        loss.backward()
        torch.cuda.synchronize()

        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            np.savez('errors_{}.pkl'.format(iter), *qconfig.grads)


def variance_profile(model_and_loss, optimizer, val_loader, prefix='.', num_batches=10000):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    # Get top 10 batches
    m.set_debug(True)
    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    data_iter = enumerate(val_loader)
    inputs = []
    targets = []
    batch_grad = None
    quant_var = None

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name : layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad

    cnt = 0
    for i, (input, target) in tqdm(data_iter):
        cnt += 1

        inputs.append(input.clone())
        targets.append(target.clone())

        # Deterministic
        qconfig.quantize_gradient = False
        mean_grad = bp(input, target)
        batch_grad = dict_add(batch_grad, mean_grad)

        if cnt == num_batches:
            break

    num_batches = cnt
    batch_grad = dict_mul(batch_grad, 1.0 / num_batches)

    def get_variance():
        total_var = None
        for i, input, target in tqdm(zip(range(num_batches), inputs, targets)):
            grad = bp(input, target)
            total_var = dict_add(total_var, dict_sqr(dict_minus(grad, batch_grad)))

        grads = [total_var[k].sum() / num_batches for k in weight_names]
        print(grads)
        return grads

    qconfig.quantize_gradient = True
    grads = [get_variance()]
    for layer in tqdm(m.linear_layers):
        layer.exact = True
        grads.append(get_variance())

    grads = np.array(grads)

    for i in range(grads.shape[0]-1):
        grads[i] -= grads[i+1]

    np.save(prefix + '/error_profile.npy', grads)
    with open(prefix + '/layer_names.pkl', 'wb') as f:
        pickle.dump(weight_names, f)

    grads = np.maximum(grads, 0)
    # grads = np.minimum(grads, 1)
    for i in range(grads.shape[0]):
        for j in range(grads.shape[1]):
            if j > i:
                grads[i, j] = 0

    fig, ax = plt.subplots(figsize=(20, 20))
    im = ax.imshow(grads, cmap='Blues', norm=LogNorm(vmin=0.01, vmax=10.0))
    ax.set_xticks(np.arange(len(weight_names)))
    ax.set_yticks(np.arange(len(weight_names)))
    ax.set_xticklabels(weight_names)
    ax.set_yticklabels(weight_names)
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")
    cbar = ax.figure.colorbar(im, ax=ax)

    for i in range(grads.shape[0]):
        for j in range(grads.shape[1]):
            text = ax.text(j, i, int(grads[i, j] * 10),
                           ha="center", va="center")

    fig.savefig('variance_profile.pdf')


def get_var(model_and_loss, optimizer, val_loader, num_batches=10000):
    if hasattr(model_and_loss.model, 'module'):
        m = model_and_loss.model.module
    else:
        m = model_and_loss.model

    # Get top 10 batches
    m.set_debug(True)
    m.set_name()
    weight_names = [layer.layer_name for layer in m.linear_layers]

    data_iter = enumerate(val_loader)
    inputs = []
    targets = []
    batch_grad = None
    quant_var = None

    def bp(input, target):
        optimizer.zero_grad()
        loss, output = model_and_loss(input, target)
        loss.backward()
        torch.cuda.synchronize()
        grad = {layer.layer_name : layer.weight.grad.detach().cpu() for layer in m.linear_layers}
        return grad

    cnt = 0
    for i, (input, target) in tqdm(data_iter):
        cnt += 1

        inputs.append(input.clone())
        targets.append(target.clone())

        # Deterministic
        qconfig.quantize_gradient = False
        mean_grad = bp(input, target)
        batch_grad = dict_add(batch_grad, mean_grad)

        if cnt == num_batches:
            break

    num_batches = cnt
    batch_grad = dict_mul(batch_grad, 1.0 / num_batches)

    def get_variance():
        total_var = None
        for i, input, target in tqdm(zip(range(num_batches), inputs, targets)):
            grad = bp(input, target)
            total_var = dict_add(total_var, dict_sqr(dict_minus(grad, batch_grad)))

        grads = [total_var[k].sum() / num_batches for k in weight_names]
        return grads

    qconfig.quantize_gradient = True
    q_grads = get_variance()
    qconfig.quantize_gradient = False
    s_grads = get_variance()

    all_qg = 0
    all_sg = 0
    for i, k in enumerate(weight_names):
        qg = q_grads[i].sum()
        sg = s_grads[i].sum()
        all_qg += qg
        all_sg += sg
        print('{}, overall var = {}, quant var = {}, sample var = {}'.format(k, qg, qg-sg, sg))

    print('Overall Var = {}, Quant Var = {}, Sample Var = {}'.format(all_qg, all_qg - all_sg, all_sg))
