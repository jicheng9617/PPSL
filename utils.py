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
    """
    Counts the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        int: Total number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_func(type: str, preference_vector: Tensor, func_value: Tensor, z: Tensor, mu: float = 0.01): 
    """
    Computes total loss by aggregating over all preference-objective pairs.
    
    Args:
        type (str): Scalarization type ('ls', 'tch', 'stch')
        preference_vector (Tensor): Preference weights λ
        func_value (Tensor): Objective values f(x)
        z (Tensor): Reference point (typically zero)
        mu (float): Temperature parameter for smooth TCH
        
    Returns:
        Tensor: Scalar loss value
    """
    agg_value = agge_func(type, preference_vector, func_value, z, mu)
    return  torch.sum(agg_value)


def agge_func(type: str, preference_vector: Tensor, func_value: Tensor, z: Tensor, mu: float = 0.01): 
    """
    Scalarization functions for multi-objective optimization.
    
    Args:
        type (str): Scalarization type
        preference_vector (Tensor): Preference weights λ ∈ Δ^m
        func_value (Tensor): Objective values f(x) ∈ R^m
        z (Tensor): Reference point
        mu (float): Smoothing parameter for STCH
        
    Returns:
        Tensor: Scalarized objective values
        
    Notes:
        - LS: Linear scalarization, g_ls = λ^T(f(x) - z)
        - TCH: Tchebycheff, g_tch = max_i{λ_i(f_i(x) - z_i)}
        - STCH: Smooth Tchebycheff, g_stch = μ·log(Σexp(λ_i(f_i(x)-z_i)/μ))
    """    
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
    """
    Estimates gradients for black-box objectives using Evolution Strategies (ES).
    
    Args:
        problem: Black-box problem instance
        x (Tensor): Current solutions
        param (Tensor): Problem parameters
        loss_type (str): Scalarization type
        preference_vecotr (Tensor): Preference vectors
        z (Tensor): Reference point
        n_grad_esti (int): Number of samples for estimation
        sigma (float): Perturbation magnitude
        
    Returns:
        Tensor: Estimated gradients ∇_x L(x)
        
    Notes:
        - Uses ranking-based ES with linear weights r_k ∈ [0.5, -0.5]
        - Gradient estimate: ∇L ≈ (1/σn)Σ r_k·u_k where u_k are random directions
        - Binary sampling: u_k ∈ {0,1}^d for efficiency
    """
    n_x, n_dim = x.shape
    device = x.device.type
    
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