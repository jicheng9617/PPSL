import torch 
from torch import Tensor

import numpy as np 
import random  


def set_seed(seed):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_func(type: str, preference_vector: Tensor, func_value: Tensor, z: Tensor, mu: float = 0.01): 
    agg_value = agge_func(type, preference_vector, func_value, z, mu)
    return  torch.sum(agg_value)


def agge_func(type: str, preference_vector: Tensor, func_value: Tensor, z: Tensor, mu: float = 0.01): 
    
    match type.lower(): 
        case 'ls': 
            agg_value = torch.sum(preference_vector * (func_value - z), dim=1)
        
        case 'tch': 
            agg_value =  torch.max(preference_vector * (func_value - z), dim=1)[0] 
        
        case 'stch': 
            agg_value = mu* torch.logsumexp(preference_vector * (func_value - z) / mu, dim=1)   
            
    return agg_value
        

@torch.no_grad()
def gradient_estimation(problem, x, param, loss_type, preference_vecotr, z, n_grad_esti, sigma: float = 0.02): 
    n_x, n_dim = x.shape
    device = x.device.type
    
    # fitness evaluation
    # tmps = torch.arange(1, n_grad_esti+1)
    # denominator = torch.max(torch.tensor(0), torch.log(torch.tensor(1+n_grad_esti/2))-torch.log(tmps)).sum()
    # r = torch.max(torch.tensor(0), torch.log(torch.tensor(1+n_grad_esti/2))-torch.log(tmps)) / denominator - 1/n_grad_esti
    r = torch.linspace(0.5, -0.5, n_grad_esti)
    
    # gradient
    grad = torch.zeros(x.shape, device=device)
    for i in range(n_x):
        x_tmp, pref_tmp = x[i], preference_vecotr[i]
        # sampled u_k
        sampled_dire = torch.bernoulli(.5 * torch.ones((n_grad_esti, n_dim), device=device))
        
        # evaluate r_k
        func_values = problem.evaluate(x_tmp + sigma*sampled_dire, param) 
        agg_value = agge_func(loss_type, pref_tmp, func_values, z)
        
        # obtain r_k
        _, sorted_indices = torch.sort(agg_value, descending=True)
        r_sorted = r[sorted_indices].reshape(-1, 1)
        
        # compute the estimated gradient
        grad[i] = torch.sum(r_sorted * sampled_dire, dim=0) / (sigma * n_grad_esti)
    
    return grad