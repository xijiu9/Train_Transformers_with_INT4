import torch
import math

def fake_quantize(h_input, h_weight, scale_input, scale_weight):
    # nomean = "Fuck"
    # assert h_input.dtype == torch.float16, "h_input must be half float datatype!"
    # assert h_weight.dtype == torch.float16, "h_weight must be half float datatype!"
    q_input = (h_input / scale_input).round().clamp(-8, 7)
    q_weight = (h_weight / scale_weight).round().clamp(-8, 7)
    
    input_q = q_input * scale_input
    weight_q = q_weight * scale_weight
    
    return input_q, weight_q

def special_quantize_grad_test_float(grad_output, q_weight, scale_weight, num_bits, lsq_input, first_transform, second_transform, x1_len, x2_len, scale1):
    num_bins_half = 2 ** num_bits - 2
    first_quantize = (first_transform * num_bins_half / scale1)
    second_quantize = (second_transform / scale1)
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    
    # leverage score
    vec_norm = torch.cat([x1_len, x2_len])
    len_norm = len(vec_norm)
        
    cnt = 0
    norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum())

    while norm_activation_loop.max() > 1 and cnt < len_norm / 2:
        small_index = (norm_activation_loop < 1)
        small_value = norm_activation_loop[small_index]
        cnt = len_norm - len(small_value)
        norm_activation_loop = torch.clamp(norm_activation_loop, 0, 1)
        if small_value.max() == 0:
            break
        small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
        norm_activation_loop[small_index] = small_value
    
    sample_index = torch.bernoulli(norm_activation_loop)
    left_indices = (sample_index != 1)
    norm_activation_loop[norm_activation_loop == 0] = 1e-10
    output_dequantize =  output_dequantize / norm_activation_loop.unsqueeze(1)
    output_dequantize[left_indices] = 0
    
    # dequantize inputx
    dequantize_sample_x = (output_dequantize[0:output_dequantize.shape[0] // 2] + output_dequantize[output_dequantize.shape[0] // 2:])
    # dequantize_sample_x = first + second
    
    # dequantize inputy
    dequantize_sample_y = q_weight * (scale_weight)
    grad_out = dequantize_sample_x.matmul(dequantize_sample_y)
    
    # calculate grad_activation and grad_scale_activation through LSQ
    # q_w = h_input / scale_input
    indicate_small = (lsq_input < -8).float()
    indicate_big = (lsq_input > 7).float()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(lsq_input.numel() * 7)
    grad_scale_input = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -lsq_input + lsq_input.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_input = indicate_middle * grad_out
    
    return grad_input, grad_scale_input

def special_quantize_grad_input_quantize(grad_output, num_bits):
    mn = min(grad_output.min() - 1e-8, 0)
    mx = max(grad_output.max() + 1e-8, 0)
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    
    # leverage score
    x1_len = torch.linalg.norm(first_transform, dim=1)
    x2_len = torch.linalg.norm(second_transform, dim=1)
    first_transform = first_transform.to(torch.int8)
    second_transform = second_transform.to(torch.int8)
    return first_transform, second_transform, x1_len, x2_len, scale1

def special_quantize_grad_input_float(grad_output, q_weight, scale_weight, num_bits, lsq_input):
    mn = min(grad_output.min() - 1e-8, 0)
    mx = max(grad_output.max() + 1e-8, 0)
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    first_quantize = (first_transform * num_bins_half / scale1)
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    second_quantize = (second_transform / scale1)
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    
    # leverage score
    x_len = torch.linalg.norm(output_dequantize, dim=1)
    vec_norm = x_len
    len_norm = len(vec_norm)
        
    cnt = 0
    norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum())

    while norm_activation_loop.max() > 1 and cnt < len_norm / 2:
        small_index = (norm_activation_loop < 1)
        small_value = norm_activation_loop[small_index]
        cnt = len_norm - len(small_value)
        norm_activation_loop = torch.clamp(norm_activation_loop, 0, 1)
        if small_value.max() == 0:
            break
        small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
        norm_activation_loop[small_index] = small_value
    
    sample_index = torch.bernoulli(norm_activation_loop)
    left_indices = (sample_index != 1)
    norm_activation_loop[norm_activation_loop == 0] = 1e-10
    output_dequantize =  output_dequantize / norm_activation_loop.unsqueeze(1)
    output_dequantize[left_indices] = 0
    
    # dequantize inputx
    dequantize_sample_x = (output_dequantize[0:output_dequantize.shape[0] // 2] + output_dequantize[output_dequantize.shape[0] // 2:])
    # dequantize_sample_x = first + second
    
    # dequantize inputy
    dequantize_sample_y = q_weight * (scale_weight)
    grad_out = dequantize_sample_x.matmul(dequantize_sample_y)
    
    # calculate grad_activation and grad_scale_activation through LSQ
    # q_w = h_input / scale_input
    indicate_small = (lsq_input < -8).float()
    indicate_big = (lsq_input > 7).float()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(lsq_input.numel() * 7)
    grad_scale_input = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -lsq_input + lsq_input.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_input = indicate_middle * grad_out
    
    return grad_input, grad_scale_input

def special_quantize_grad_input(grad_output, q_weight, scale_weight, num_bits, lsq_input):
    assert grad_output.dtype == torch.float16, "grad_output must be half float datatype!"
    # assert dequantize_weight.dtype == torch.float16, "dequantize_weight must be half float datatype!"
    # assert h_input.dtype == torch.float16, "h_input must be half float datatype!"
    mn = min(grad_output.min() - 1e-8, 0).float()
    mx = max(grad_output.max() + 1e-8, 0).float()
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output.float() * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    first_quantize = (first_transform * num_bins_half / scale1).half()
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    second_quantize = (second_transform / scale1).half()
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    
    # leverage score
    x_len = torch.linalg.norm(output_dequantize, dim=1)
    vec_norm = x_len.float()
    len_norm = len(vec_norm)
        
    cnt = 0
    norm_activation_loop = vec_norm * len_norm / (2 * vec_norm.sum())

    while norm_activation_loop.max() > 1 and cnt < len_norm / 2:
        small_index = (norm_activation_loop < 1)
        small_value = norm_activation_loop[small_index]
        cnt = len_norm - len(small_value)
        norm_activation_loop = torch.clamp(norm_activation_loop, 0, 1)
        if small_value.max() == 0:
            break
        small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
        norm_activation_loop[small_index] = small_value
    
    sample_index = torch.bernoulli(norm_activation_loop)
    left_indices = (sample_index != 1)
    norm_activation_loop[norm_activation_loop == 0] = 1e-10
    output_dequantize =  output_dequantize / norm_activation_loop.unsqueeze(1)
    output_dequantize[left_indices] = 0
    
    # dequantize inputx
    dequantize_sample_x = (output_dequantize[0:output_dequantize.shape[0] // 2] + output_dequantize[output_dequantize.shape[0] // 2:]).half()
    # dequantize_sample_x = first + second
    
    # dequantize inputy
    dequantize_sample_y = q_weight * (scale_weight.half())
    grad_out = dequantize_sample_x.matmul(dequantize_sample_y)
    
    # calculate grad_activation and grad_scale_activation through LSQ
    # q_w = h_input / scale_input
    indicate_small = (lsq_input < -8).half()
    indicate_big = (lsq_input > 7).half()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(lsq_input.numel() * 7)
    grad_scale_input = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -lsq_input + lsq_input.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_input = indicate_middle * grad_out
    
    return grad_input, grad_scale_input

def special_quantize_grad_weight_test(first_transform, second_transform, x1_len, x2_len, scale1, q_input, scale_input, num_bits, lsq_w):
    # assert grad_output.dtype == torch.float16, "grad_output must be half float datatype!"
    # assert dequantize_input.dtype == torch.float16, "dequantize_input must be half float datatype!"
    num_bins_half = 2 ** num_bits - 2
    first_quantize = (first_transform * num_bins_half / scale1)
    
    second_quantize = (second_transform / scale1)
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    
    dequantize_input = q_input * scale_input
    y2 = torch.cat([dequantize_input, dequantize_input], 0)
    x_len = torch.cat([x1_len, x2_len], dim=0)
    y_len = torch.linalg.norm(dequantize_input, dim=1)
    
    vec_norm = x_len.mul(y_len)
    len_norm = len(vec_norm)
    
    cnt = 0
    norm_weight_loop = vec_norm * len_norm / (2 * vec_norm.sum())
    norm_weight_loop[norm_weight_loop > 0] = 1

    sample_index = norm_weight_loop
    index = torch.nonzero((sample_index == 1)).squeeze(1)
    norm_weight_loop[norm_weight_loop == 0] = 1e-10
    output_dequantize = output_dequantize / norm_weight_loop.unsqueeze(1)
    
    sample_x = output_dequantize[index, :]
    sample_y = y2[index, :]
    
    # dequantize inputx    
    dequantize_sample_x = sample_x
    
    # dequantize inputy
    dequantize_sample_y = sample_y 
    grad_out = dequantize_sample_x.t().matmul(dequantize_sample_y)
    
    indicate_small = (lsq_w < -8).float()
    indicate_big = (lsq_w > 7).float()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(lsq_w.numel() * 7)
    grad_scale_weight = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -lsq_w + lsq_w.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_weight = indicate_middle * grad_out
    
    return grad_weight, grad_scale_weight

def special_quantize_grad_weight_float(grad_output, q_input, scale_input, num_bits, lsq_w):
    # assert grad_output.dtype == torch.float16, "grad_output must be half float datatype!"
    # assert dequantize_input.dtype == torch.float16, "dequantize_input must be half float datatype!"
    mn = min(grad_output.min() - 1e-8, 0)
    mx = max(grad_output.max() + 1e-8, 0)
    # print("max is:",max(abs(mn), abs(mx)))
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    first_quantize = (first_transform * num_bins_half / scale1)
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    second_quantize = (second_transform / scale1)
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    x1_len = torch.linalg.norm(first_quantize, dim=1)
    x2_len = torch.linalg.norm(second_quantize, dim=1)
    
    dequantize_input = q_input * scale_input
    y2 = torch.cat([dequantize_input, dequantize_input], 0)
    x_len = torch.cat([x1_len, x2_len], dim=0)
    y_len = torch.linalg.norm(y2, dim=1)
    
    vec_norm = x_len.mul(y_len)
    len_norm = len(vec_norm)
    
    cnt = 0
    norm_weight_loop = vec_norm * len_norm / (2 * vec_norm.sum())
    norm_weight_loop[norm_weight_loop > 0] = 1

    # while norm_weight_loop.max() > 1 and cnt < len_norm / 2:
    #     small_index = (norm_weight_loop < 1)
    #     small_value = norm_weight_loop[small_index]
    #     cnt = len_norm - len(small_value)
    #     norm_weight_loop = torch.clamp(norm_weight_loop, 0, 1)
    #     if small_value.max() == 0 :
    #         break
    #     small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
    #     norm_weight_loop[small_index] = small_value

    # sample_index = torch.bernoulli(norm_weight_loop)
    sample_index = norm_weight_loop
    index = torch.nonzero((sample_index == 1)).squeeze(1)
    norm_weight_loop[norm_weight_loop == 0] = 1e-10
    output_dequantize = output_dequantize / norm_weight_loop.unsqueeze(1)
    
    sample_x = output_dequantize[index, :]
    sample_y = y2[index, :]
    
    # dequantize inputx    
    dequantize_sample_x = sample_x
    
    # dequantize inputy
    dequantize_sample_y = sample_y 
    grad_out = dequantize_sample_x.t().matmul(dequantize_sample_y)
    
    indicate_small = (lsq_w < -8).float()
    indicate_big = (lsq_w > 7).float()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(lsq_w.numel() * 7)
    grad_scale_weight = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -lsq_w + lsq_w.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_weight = indicate_middle * grad_out
    
    return grad_weight, grad_scale_weight, first_transform.to(torch.int8), second_transform.to(torch.int8), x1_len, x2_len, scale1

def special_quantize_grad_weight(grad_output, q_input, scale_input, num_bits, lsq_w):
    assert grad_output.dtype == torch.float16, "grad_output must be half float datatype!"
    # assert dequantize_input.dtype == torch.float16, "dequantize_input must be half float datatype!"
    mn = min(grad_output.min() - 1e-8, 0).float()
    mx = max(grad_output.max() + 1e-8, 0).float()
    print("max is:",max(abs(mn), abs(mx)))
    
    num_bins_half = 2 ** num_bits - 2
    num_bins = num_bins_half * num_bins_half
    
    scale1 = num_bins / (2 * max(mn.abs(), mx.abs()))
    
    first_transform_0 = (grad_output.float() * scale1)
    first_transform = (first_transform_0 / num_bins_half).clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    first_quantize = (first_transform * num_bins_half / scale1).half()
    
    second_transform = first_transform_0 - first_transform * num_bins_half
    noise = second_transform.new(second_transform.shape).uniform_(-0.5, 0.5)
    second_transform.add_(noise)
    second_transform.clamp_(1 - num_bins_half // 2, num_bins_half // 2 - 1).round_()
    second_quantize = (second_transform / scale1).half()
    output_dequantize = torch.cat([first_quantize, second_quantize], dim=0)
    x1_len = torch.linalg.norm(first_quantize, dim=1)
    x2_len = torch.linalg.norm(second_quantize, dim=1)
    
    dequantize_input = q_input * (scale_input.half())
    y2 = torch.cat([dequantize_input, dequantize_input], 0)
    x_len = torch.cat([x1_len, x2_len], dim=0)
    y_len = torch.linalg.norm(y2, dim=1)
    
    vec_norm = x_len.mul(y_len).float()
    len_norm = len(vec_norm)
    
    cnt = 0
    norm_weight_loop = vec_norm * len_norm / (2 * vec_norm.sum())

    while norm_weight_loop.max() > 1 and cnt < len_norm / 2:
        small_index = (norm_weight_loop < 1)
        small_value = norm_weight_loop[small_index]
        cnt = len_norm - len(small_value)
        norm_weight_loop = torch.clamp(norm_weight_loop, 0, 1)
        if small_value.max() == 0 :
            break
        small_value = small_value * (len_norm // 2 - cnt) / small_value.sum()
        norm_weight_loop[small_index] = small_value

    if torch.isnan(norm_weight_loop).sum() > 0:
        import IPython
        IPython.embed()
    sample_index = torch.bernoulli(norm_weight_loop)
    index = torch.nonzero((sample_index == 1)).squeeze(1)
    norm_weight_loop[norm_weight_loop == 0] = 1e-10
    output_dequantize = output_dequantize / norm_weight_loop.unsqueeze(1)
    
    sample_x = output_dequantize[index, :]
    sample_y = y2[index, :]
    
    # dequantize inputx    
    dequantize_sample_x = sample_x.half()
    
    # dequantize inputy
    dequantize_sample_y = sample_y 
    grad_out = dequantize_sample_x.t().matmul(dequantize_sample_y)
    
    indicate_small = (lsq_w < -8).half()
    indicate_big = (lsq_w > 7).half()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(lsq_w.numel() * 7)
    grad_scale_weight = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -lsq_w + lsq_w.round())) * grad_out * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_weight = indicate_middle * grad_out
    
    return grad_weight, grad_scale_weight, first_transform.to(torch.int8), second_transform.to(torch.int8), x1_len, x2_len, scale1

def find_mask(tens):
    max_val = torch.max(tens)
    threshold = max_val / 10.0
    bool_idx = tens > threshold

    return bool_idx

def find_mask_weight(tens):
    max_val = torch.max(tens)
    threshold = max_val / 10.0
    bool_idx = tens > threshold
    bool_num = len(tens) - bool_idx.sum()
    bool_num2 = ((bool_num/32).floor() * 32).int()
    if bool_num2 < bool_num:
        indices = torch.where(~bool_idx)[0][:bool_num-bool_num2]
        bool_idx[indices] = 1
        
    return bool_idx

def quantize_grad(grad_output_flatten, q_input_flatten, q_weight, scale_input, scale_weight, num_bits):
    norm1 = torch.norm(grad_output_flatten, p=1, dim=1)
    norm2 = torch.norm(grad_output_flatten, p=2, dim=1)
    input_flatten = q_input_flatten * scale_input
    norm_square = torch.norm(input_flatten, p=2, dim=1) ** 2
    leverage_weight = norm1 * norm_square
    leverage_input = norm2
    
    # leverage_mask_weight = find_mask(leverage_weight)
    leverage_mask_weight = find_mask_weight(leverage_weight)
    # import IPython
    # IPython.embed()
    leverage_mask_input = find_mask(leverage_input)
    num_bins_half = 2 ** num_bits - 2
    
    # quantize weight
    grad_weight_flatten = grad_output_flatten[~leverage_mask_weight]
    # assert((~leverage_mask_weight).sum() % 32 == 0)
    if grad_weight_flatten.numel() > 0:
        with torch.no_grad():
            mn = min(grad_weight_flatten.min() - 1e-8, 0)
            mx = max(grad_weight_flatten.max() + 1e-8, 0)
            
            scale = num_bins_half / (2 * max(mn.abs(), mx.abs()))
            
            Qn_weight = -torch.ones_like(grad_weight_flatten) * (num_bins_half // 2)
            Qp_weight = torch.ones_like(grad_weight_flatten) * (num_bins_half // 2)
            q_grad_weight_flatten = grad_weight_flatten * scale
            
            noise_weight = q_grad_weight_flatten.new(q_grad_weight_flatten.shape).uniform_(-0.5, 0.5)
            q_grad_weight_flatten.add_(noise_weight)
            q_grad_weight_flatten.clamp_(Qn_weight, Qp_weight).round_()
        
        grad_weight_LSS = q_grad_weight_flatten.float().t().mm(q_input_flatten[~leverage_mask_weight].float()) / scale * scale_input \
                + grad_output_flatten[leverage_mask_weight].t().mm(q_input_flatten[leverage_mask_weight]*scale_input)
    
    else:
        grad_weight_LSS = grad_output_flatten[leverage_mask_weight].t().mm(q_input_flatten[leverage_mask_weight]*scale_input)
        
    # quantize input
    grad_input_flatten = grad_output_flatten[~leverage_mask_input]
    grad_input_LSS = torch.empty(grad_output_flatten.shape[0], q_weight.shape[-1]).cuda()
    if grad_input_flatten.numel() > 0:
        with torch.no_grad():
            mn_vec = torch.minimum(grad_input_flatten.min(dim=1)[0] - 1e-8, torch.zeros_like(grad_input_flatten.min(dim=1)[0]))
            mx_vec = torch.maximum(grad_input_flatten.max(dim=1)[0] + 1e-8, torch.zeros_like(grad_input_flatten.max(dim=1)[0]))
            
            scale_vec = num_bins_half / (2 * torch.maximum(mn_vec.abs(), mx_vec.abs())).unsqueeze(-1)
            
            Qn_input = -torch.ones_like(grad_input_flatten) * (num_bins_half // 2)
            Qp_input = torch.ones_like(grad_input_flatten) * (num_bins_half // 2)
            q_grad_input_flatten = grad_input_flatten * scale_vec
            
            noise_input = q_grad_input_flatten.new(q_grad_input_flatten.shape).uniform_(-0.5, 0.5)
            q_grad_input_flatten.add_(noise_input)
            q_grad_input_flatten.clamp_(Qn_input, Qp_input).round_()
        
        # import IPython
        # IPython.embed()
        
        grad_input_LSS[leverage_mask_input] = grad_output_flatten[leverage_mask_input].mm(q_weight*scale_weight)
        grad_input_LSS[~leverage_mask_input] = q_grad_input_flatten.float().mm(q_weight.float()) / scale_vec * scale_weight
        # grad_input_LSS = q_grad_input_flatten.float().mm(q_weight.float()) / scale_vec * scale_weight \
        #         + grad_output_flatten[leverage_mask_input].mm(q_weight*scale_weight)
    else:
        grad_input_LSS[leverage_mask_input] = grad_output_flatten[leverage_mask_input].mm(q_weight*scale_weight)
        grad_input_LSS[~leverage_mask_input] = 0
        
    # calculate LSS weight & input
                    
    # calculate LSQ weight & input
    indicate_small = (q_weight < -8).float()
    indicate_big = (q_weight > 7).float()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(q_weight.numel() * 7)
    grad_scale_weight = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -q_weight + q_weight.float().round())) * grad_weight_LSS * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_weight = indicate_middle * grad_weight_LSS
    
    indicate_small = (q_input_flatten < -8).float()
    indicate_big = (q_input_flatten > 7).float()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(q_input_flatten.numel() * 7)
    grad_scale_input = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -q_input_flatten + q_input_flatten.float().round())) * grad_input_LSS * grad_scale).sum().unsqueeze(dim=0)
    #Todo:need to matmul a hadamard matrix?
    grad_input = indicate_middle * grad_input_LSS
    
    return grad_input, grad_weight, grad_scale_input, grad_scale_weight

def quantize_grad_half(grad_output_flatten, q_input_flatten, q_weight, scale_input, scale_weight, num_bits):
    norm1 = torch.norm(grad_output_flatten, p=1, dim=1).float()
    norm2 = torch.norm(grad_output_flatten, p=2, dim=1).float()
    input_flatten = q_input_flatten * scale_input
    norm_square = (torch.norm(input_flatten, p=2, dim=1) ** 2).float()
    leverage_weight = norm1 * norm_square
    leverage_input = norm2
    
    # leverage_mask_weight = find_mask(leverage_weight)
    leverage_mask_weight = find_mask_weight(leverage_weight)
    # import IPython
    # IPython.embed()
    leverage_mask_input = find_mask(leverage_input)
    num_bins_half = 2 ** num_bits - 2
    
    # quantize weight
    grad_weight_flatten = grad_output_flatten[~leverage_mask_weight]
    # assert((~leverage_mask_weight).sum() % 32 == 0)
    if grad_weight_flatten.numel() > 0:
        with torch.no_grad():
            mn = min(grad_weight_flatten.min().float()  - 1e-8, 0)
            mx = max(grad_weight_flatten.max().float() + 1e-8, 0)
            
            scale = num_bins_half / (2 * max(mn.abs(), mx.abs()))
            
            Qn_weight = -torch.ones_like(grad_weight_flatten) * (num_bins_half // 2)
            Qp_weight = torch.ones_like(grad_weight_flatten) * (num_bins_half // 2)
            q_grad_weight_flatten = (grad_weight_flatten.float() * scale).half()
            
            noise_weight = q_grad_weight_flatten.new(q_grad_weight_flatten.shape).uniform_(-0.5, 0.5)
            q_grad_weight_flatten.add_(noise_weight)
            q_grad_weight_flatten.clamp_(Qn_weight, Qp_weight).round_()
        
        grad_weight_LSS = (q_grad_weight_flatten.half().t().mm(q_input_flatten[~leverage_mask_weight].half()) / scale * scale_input).half() \
                + grad_output_flatten[leverage_mask_weight].t().mm(q_input_flatten[leverage_mask_weight]*scale_input)
    
    else:
        grad_weight_LSS = grad_output_flatten[leverage_mask_weight].t().mm(q_input_flatten[leverage_mask_weight]*scale_input)
        
    # quantize input
    grad_input_flatten = grad_output_flatten[~leverage_mask_input]
    grad_input_LSS = torch.empty(grad_output_flatten.shape[0], q_weight.shape[-1]).cuda().half()
    if grad_input_flatten.numel() > 0:
        with torch.no_grad():
            mn_vec = torch.minimum(grad_input_flatten.min(dim=1)[0].float() - 1e-8, torch.zeros_like(grad_input_flatten.min(dim=1)[0]))
            mx_vec = torch.maximum(grad_input_flatten.max(dim=1)[0].float() + 1e-8, torch.zeros_like(grad_input_flatten.max(dim=1)[0]))
            
            scale_vec = num_bins_half / (2 * torch.maximum(mn_vec.abs(), mx_vec.abs())).unsqueeze(-1)
            
            Qn_input = -torch.ones_like(grad_input_flatten) * (num_bins_half // 2)
            Qp_input = torch.ones_like(grad_input_flatten) * (num_bins_half // 2)
            q_grad_input_flatten = (grad_input_flatten * scale_vec).half()
            
            noise_input = q_grad_input_flatten.new(q_grad_input_flatten.shape).uniform_(-0.5, 0.5)
            q_grad_input_flatten.add_(noise_input)
            q_grad_input_flatten.clamp_(Qn_input, Qp_input).round_()
        
        # import IPython
        # IPython.embed()
        
        grad_input_LSS[leverage_mask_input] = grad_output_flatten[leverage_mask_input].mm(q_weight*scale_weight)
        grad_input_LSS[~leverage_mask_input] = (q_grad_input_flatten.half().mm(q_weight.half()) / scale_vec * scale_weight).half()
        # grad_input_LSS = q_grad_input_flatten.float().mm(q_weight.float()) / scale_vec * scale_weight \
        #         + grad_output_flatten[leverage_mask_input].mm(q_weight*scale_weight)
    else:
        grad_input_LSS[leverage_mask_input] = grad_output_flatten[leverage_mask_input].mm(q_weight*scale_weight)
        grad_input_LSS[~leverage_mask_input] = 0
        
    # calculate LSS weight & input
                    
    # calculate LSQ weight & input
    indicate_small = (q_weight < -8).half()
    indicate_big = (q_weight > 7).half()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(q_weight.numel() * 7)
    grad_scale_weight = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -q_weight + q_weight.half().round())) * grad_weight_LSS * grad_scale).sum().unsqueeze(dim=0).half()
    #Todo:need to matmul a hadamard matrix?
    grad_weight = indicate_middle * grad_weight_LSS
    
    indicate_small = (q_input_flatten < -8).half()
    indicate_big = (q_input_flatten > 7).half()
    indicate_middle = 1.0 - indicate_small - indicate_big
    grad_scale = 1.0 / math.sqrt(q_input_flatten.numel() * 7)
    grad_scale_input = ((indicate_small * -8 + indicate_big * 7 + indicate_middle * (
            -q_input_flatten + q_input_flatten.half().round())) * grad_input_LSS * grad_scale).sum().unsqueeze(dim=0).half()
    #Todo:need to matmul a hadamard matrix?
    grad_input = indicate_middle * grad_input_LSS
    
    # if torch.count_nonzero(torch.isnan(grad_input)) != 0 or torch.count_nonzero(torch.isnan(grad_weight)) != 0:
    #     import IPython
    #     IPython.embed()
    return grad_input, grad_weight, grad_scale_input, grad_scale_weight