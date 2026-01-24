import torch
from torch import nn
import random
import numpy as np 
import logging

def generate_layer_density_dict(layer_shape_dict, num_overall_elements, sparse_layer_set, target_density, layer_density_strategy):
    # the maximum number of elements
    num_remain_elements = int(target_density * num_overall_elements)

    # the number of elements in the dense layer and sparse layer
    num_dense_elements = 0
    for name, shape in layer_shape_dict.items():
        if name not in sparse_layer_set:
            num_dense_elements += np.prod(shape)

    assert num_remain_elements > num_dense_elements, f"the number of elements({num_dense_elements}) left in dense model is higher than minimum elements  requirement ({num_remain_elements}) under target density {target_density}. Please use higher target density or fewer ignore dense layers "

    num_remain_sparse_elements = num_remain_elements - num_dense_elements

    layer_density_dict = {}
    if layer_density_strategy == "uniform":
        layer_wise_density = num_remain_sparse_elements/(num_overall_elements - num_dense_elements)

        for name, shape in layer_shape_dict.items():
            if name in sparse_layer_set:
                assert int(np.prod(shape) * layer_wise_density) >= 1 , f"the layer wise density {layer_wise_density} is so small that make {name} to be empty"

                layer_density_dict[name] = layer_wise_density

    elif layer_density_strategy == "ER":
        real_density  = num_remain_sparse_elements/(num_overall_elements - num_dense_elements)
        layer_density_dict = get_erdos_renyi_dist(layer_shape_dict, sparse_layer_set, real_density, False)
        
    elif layer_density_strategy == "ERK":
        real_density  = num_remain_sparse_elements/(num_overall_elements - num_dense_elements)
        layer_density_dict = get_erdos_renyi_dist(layer_shape_dict, sparse_layer_set, real_density, True)

    else:
        raise Exception(f"layer density strategy {layer_density_strategy} is not supported")

    return layer_density_dict


def pruning(model, layer_density_dict, pruning_strategy, prune_scores=None, mask_dict=None):
    if mask_dict is None:
        mask_dict = {}

    new_mask_dict = {}

    # dw_keys = {
    #     "model." + (mn[6:] if mn.startswith("model.") else mn)
    #     for mn, m in model.named_modules()
    #     if isinstance(m, nn.Conv2d) and m.groups == m.in_channels and m.out_channels == m.in_channels
    # }

    for name, weight in model.named_parameters():
        if name in layer_density_dict:
            density = layer_density_dict[name]
            num_elements = weight.numel()

            if name not in mask_dict:
                old_mask = torch.ones_like(weight, dtype=weight.data.dtype, requires_grad=False)
            else:
                old_mask = mask_dict[name]

            if pruning_strategy in ["score"]:
                key = "model." + name.removesuffix(".weight")

                # if key in dw_keys:
                #     logging.info(f"DEPTHWISE CONV KEYS {key}")

                #     new_mask_dict[name] = old_mask
                if prune_scores is not None and key in prune_scores:
                    logging.info(f"ENTER SCORE PRUNE WITH KEY {key}")

                    new_mask_dict[name] = score_prune(prune_scores[key], old_mask, num_elements, density)
                else:
                    logging.info(f"KEY HAS NO SCORE {key}")

                    new_mask_dict[name] = old_mask
                    # new_mask_dict[name] = random_prune(weight, old_mask, num_elements, density)

            elif pruning_strategy in ["mag", "magnitude"]:
                new_mask_dict[name] = magnitude_prune(weight, old_mask, num_elements, density)
            elif pruning_strategy in ["random"]:
                new_mask_dict[name] = random_prune(old_mask, num_elements, density)
            elif pruning_strategy in ["structure-mag"]:
                pass
            else:
                raise Exception(f"pruning strategy {pruning_strategy} is not supported")

    return new_mask_dict


def score_prune(score, old_mask, num_elements, density):  # 服务端 pruning
    score = score.to(old_mask.device)

    num_remain = int(num_elements * density)
    current_num_element = old_mask.sum()
    assert current_num_element >= num_remain

    s = score.contiguous().view(-1)
    m = old_mask.contiguous().view(-1)

    idx = (m.data == 1).nonzero(as_tuple=True)[0]
    active_scores = s.index_select(0, idx)

    _, order = torch.sort(active_scores.data, descending=True)
    keep = idx[order[:num_remain]]

    new_mask = torch.zeros_like(old_mask, dtype=old_mask.dtype, requires_grad=False)
    new_mask.data.view(-1)[keep] = 1.0

    return new_mask

def magnitude_prune(weight, old_mask, num_elements, density):
    try:
        weight = weight * old_mask
    except RuntimeError:
        weight = weight * old_mask.to(weight.device) 
    
    num_remain = int(num_elements * density)
    assert old_mask.sum() >= num_remain

    x, idx = torch.sort(torch.abs(weight.data.view(-1)), descending=True)
    new_mask = torch.zeros_like( old_mask, dtype=old_mask.data.dtype, requires_grad=False )
    new_mask.data.view(-1)[idx[:num_remain]] = 1.0
    return new_mask

def random_prune(weight, old_mask, num_elements, density):
    old_mask = old_mask.to(weight.device)

    weight = weight * old_mask
    num_remain = int(num_elements * density)
    current_num_element = old_mask.sum()
    assert current_num_element >= num_remain

    idx = (old_mask.data.view(-1) == 1).nonzero(as_tuple=True)[0].tolist()
    random.shuffle(idx)
    new_mask = torch.zeros_like( old_mask, dtype=old_mask.data.dtype, requires_grad=False )
    new_mask.data.view(-1)[idx[:num_remain]] = 1.0
    return new_mask


def f_decay(t, alpha, T_end):
    # return int(alpha * (1 - t / T_end))
    return alpha / 2 * (1 + np.cos(t * np.pi / T_end))


def growing(model, mask_dict, growth_percentage):
    grad_dict = {}
    for name, param in model.named_parameters():
        if param.grad is not None and name in mask_dict:
            grad_dict[name] = param.grad.abs().view(-1)
    
    for name, mask in mask_dict.items():
        if name in grad_dict:
            num_elements = mask.numel()
            num_grow = int(num_elements * growth_percentage)

            grad = grad_dict[name]
            inactive_indices = (mask.view(-1) == 0).nonzero(as_tuple=False).view(-1)
            grad_inactive = grad[inactive_indices]

            _, topk_indices = torch.topk(grad_inactive, num_grow, sorted=False)
            new_growth_indices = inactive_indices[topk_indices]

            mask.view(-1)[new_growth_indices] = 1.0

    return mask_dict


# 
def sparse_update_step(model, gradients, mask_dict, t, T_end, alpha, scores=None):

    for name, param in model.named_parameters():
        if name in mask_dict:
            # mask_dict[name] = mask_dict[name].to(param.device)

            active_num = (mask_dict[name] == 1).int().sum().item()
            k = int(f_decay(t, alpha, T_end) * active_num)

            # pruning
            active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1)

            if active_indices.numel() > 0 and k > 0:
                flat_w = param.data.view(-1)

                active_indices = active_indices.to(flat_w.device)

                if scores is not None and name.rsplit(".", 1)[0] in scores["prune"]:
                    score_active = scores["prune"][name.rsplit(".", 1)[0]].to(param.device).view(-1).index_select(0, active_indices)
                else:
                    score_active = torch.abs(flat_w).index_select(0, active_indices)

                _, prune_local = torch.topk(score_active, min(k, score_active.numel()), largest=False)

                mask_dict[name] = mask_dict[name].to(active_indices.device)
                mask_dict[name].view(-1)[active_indices[prune_local]] = 0

            # growing
            inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1)

            if inactive_indices.numel() > 0 and k > 0:
                g_flat = gradients[name].abs().view(-1)

                inactive_indices = inactive_indices.to(g_flat.device)

                if scores is not None and name.rsplit(".", 1)[0] in scores["grow"]:
                    score_inactive = scores["grow"][name.rsplit(".", 1)[0]].to(param.device).view(-1).index_select(0, inactive_indices)
                else:
                    score_inactive = g_flat.index_select(0, inactive_indices)

                _, grow_local = torch.topk(
                    score_inactive, min(k, score_inactive.numel()),
                    largest=True, sorted=False
                )
                mask_dict[name] = mask_dict[name].to(inactive_indices.device)
                mask_dict[name].view(-1)[inactive_indices[grow_local]] = 1

    return mask_dict


def sparse_pruning_step(model, mask_dict, t, T_end, alpha):
   for name, param in model.named_parameters():
       if name in mask_dict:
           active_num = (mask_dict[name] == 1).int().sum().item()
           k = int(f_decay(t, alpha, T_end) * active_num)
           # pruning：Find the k  smallest connections among the current active connections and set them to non-active
           active_indices = (mask_dict[name].view(-1) == 1).nonzero(as_tuple=False).view(-1).cpu()
           _, prune_indices = torch.topk(torch.abs(param.data.view(-1)[active_indices]), k, largest=False)
           mask_dict[name].view(-1)[active_indices[prune_indices.cpu()]] = 0
   return mask_dict

def sparse_growing_step(model, gradients, mask_dict, layer_density_dict):
   for name, weight in model.named_parameters():
       if name in layer_density_dict:
           active_num = (mask_dict[name] == 1).int().sum().item()
           density = layer_density_dict[name]
           num_elements = weight.numel() # the total number for elements
           num_remain = int(num_elements * density)
           k = num_remain - active_num
           # growing：Find the k  largest gradients connections among the currently inactive connections and set them to active
           inactive_indices = (mask_dict[name].view(-1) == 0).nonzero(as_tuple=False).view(-1).cpu()
           grad_inactive = gradients[name].abs().view(-1)[inactive_indices].cpu()
           _, grow_indices = torch.topk(grad_inactive, k, sorted=False)
           mask_dict[name].view(-1)[inactive_indices[grow_indices.cpu()]] = 1
   return mask_dict


def get_erdos_renyi_dist(layer_shape_dict, sparse_layer_set, target_density, is_kernel: bool = True) :
    """
    Get layer-wise densities distributed according to
    ER or ERK (erdos-renyi or erdos-renyi-kernel).

    Ensures resulting densities do not cross 1
    for any layer.

    :param masking: Masking instance
    :param is_kernel: use ERK (True), ER (False)
    :return: Layer-wise density dict
    """
    # Same as Erdos Renyi with modification for conv
    # initialization used in sparse evolutionary training
    # scales the number of non-zero weights linearly proportional
    # to the product of all dimensions, that is input*output
    # for fully connected layers, and h*w*in_c*out_c for conv
    # layers.
    _erk_power_scale = 1.0

    epsilon = 1.0
    is_epsilon_valid = False
    # # The following loop will terminate worst case when all masks are in the
    # custom_sparsity_map. This should probably never happen though, since once
    # we have a single variable or more with the same constant, we have a valid
    # epsilon. Note that for each iteration we add at least one variable to the
    # custom_sparsity_map and therefore this while loop should terminate.
    _dense_layers = set()
    while not is_epsilon_valid:
        # We will start with all layers and try to find right epsilon. However if
        # any probablity exceeds 1, we will make that layer dense and repeat the
        # process (finding epsilon) with the non-dense layers.
        # We want the total number of connections to be the same. Let say we have
        # for layers with N_1, ..., N_4 parameters each. Let say after some
        # iterations probability of some dense layers (3, 4) exceeded 1 and
        # therefore we added them to the dense_layers set. Those layers will not
        # scale with erdos_renyi, however we need to count them so that target
        # paratemeter count is achieved. See below.
        # eps * (p_1 * N_1 + p_2 * N_2) + (N_3 + N_4) =
        #    (1 - default_sparsity) * (N_1 + N_2 + N_3 + N_4)
        # eps * (p_1 * N_1 + p_2 * N_2) =
        #    (1 - default_sparsity) * (N_1 + N_2) - default_sparsity * (N_3 + N_4)
        # eps = rhs / (\sum_i p_i * N_i) = rhs / divisor.

        divisor = 0
        rhs = 0
        raw_probabilities = {}
        for name, shape in layer_shape_dict.items():
            if name not in sparse_layer_set:
                continue
            n_param = np.prod(shape)
            n_zeros = int(n_param * (1 - target_density))
            n_ones = int(n_param * target_density)

            if name in _dense_layers:
                # See `- default_sparsity * (N_3 + N_4)` part of the equation above.
                rhs -= n_zeros
            else:
                # Corresponds to `(1 - default_sparsity) * (N_1 + N_2)` part of the
                # equation above.
                rhs += n_ones

                if is_kernel:
                    # Erdos-Renyi probability: epsilon * (n_in + n_out / n_in * n_out).
                    raw_probabilities[name] = (
                        np.sum(shape) / n_param
                    ) ** _erk_power_scale
                    # Note that raw_probabilities[mask] * n_param gives the individual
                    # elements of the divisor.
                else:
                    # Cin and Cout for a conv kernel
                    n_in, n_out = shape[:2]
                    raw_probabilities[name] = (n_in + n_out) / (n_in * n_out)
                divisor += raw_probabilities[name] * n_param
        # By multipliying individual probabilites with epsilon, we should get the
        # number of parameters per layer correctly.
        epsilon = rhs / divisor
        # If epsilon * raw_probabilities[mask.name] > 1. We set the sparsities of that
        # mask to 0., so they become part of dense_layers sets.
        max_prob = np.max(list(raw_probabilities.values()))
        max_prob_one = max_prob * epsilon
        if max_prob_one > 1:
            is_epsilon_valid = False
            for mask_name, mask_raw_prob in raw_probabilities.items():
                if mask_raw_prob == max_prob:
                    logging.info(f"Density of layer:{mask_name} set to 1.0")
                    _dense_layers.add(mask_name)
        else:
            is_epsilon_valid = True

    prob_dict = {}
    # With the valid epsilon, we can set sparsities of the remaning layers.
    for name in sparse_layer_set:
        if name not in _dense_layers:
            prob = min(epsilon * raw_probabilities[name], 1.)
        else:
            prob = 1.0

        prob_dict[name] = prob

    return prob_dict
