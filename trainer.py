import logging
import torch

from tqdm import trange

from model import PSModel, PSModelHyper, PSModelLoRA, PSModelLoRAHyper, PSbaseModel
from utils import *


@torch.no_grad()
def generate_ps(
    problem: any, 
    param: torch.tensor,
    hypernet: any, 
    psmodel: any, 
    n_samples: int, 
    device: torch.device,
):
    """
    Generates Pareto set solutions for a given parameter using trained PPSL models.
    
    Args:
        problem (any): Parametric multi-objective optimization problem
        param (torch.tensor): Problem parameter vector
        hypernet (any): Trained hypernetwork that maps parameters to PS model weights
        psmodel (any): Trained PS model that maps preferences to solutions
        n_samples (int): Number of Pareto solutions to generate
        device (torch.device): Computing device
        
    Returns:
        torch.tensor: Pareto set solutions of shape (n_samples, n_dim)
        
    Notes:
        - Inference only (no gradient computation)
        - Preferences sampled uniformly from simplex via Dirichlet(α=[1,...,1])
    """
    weights = hypernet(param) 
    n_obj = problem.n_obj

    alpha = torch.ones(n_obj, device=device)
    pref = torch.distributions.Dirichlet(alpha).sample((n_samples,))
    pref_vec = pref

    xs = psmodel(pref_vec, weights) 
    return xs


def trainer_hpn(
        problem: any, 
        hpn_hidden_size: int,
        psm_hidden_size: int, 
        psm_n_layer: int, 
        n_epochs: int, 
        lr: float,
        loss_type: str, 
        device: torch.device,
        lora_type: bool = False, 
        free_rank: int = 3, 
        n_sample_params: int = 5, 
        n_sample_pref: int = 30, 
):
    """
    Trains a hypernetwork-based Parametric Pareto Set Learning (PPSL) model.
    
    This function implements the core training procedure for PPSL, where a hypernetwork
    learns to generate weights for a Pareto Set (PS) model conditioned on problem parameters.
    The PS model then maps preference vectors to Pareto optimal solutions for the given
    parameter setting, enabling amortized optimization across the parameter space.
    
    Args:
        problem (any): The parametric multi-objective optimization problem instance.
                      Must have attributes: n_params, n_dim, n_obj, pu, pl, ideal_point, nadir_point
        hpn_hidden_size (int): Hidden layer size for the hypernetwork
        psm_hidden_size (int): Hidden layer size for the Pareto Set model
        psm_n_layer (int): Number of layers in the Pareto Set model
        n_epochs (int): Number of training epochs
        lr (float): Learning rate for the hypernetwork optimizer
        loss_type (str): Type of scalarization function ('ls', 'tch', 'stch', 'mtch', 'pbi', etc.)
        device (torch.device): Device to run training on (CPU/GPU)
        lora_type (bool, optional): Whether to use Low-Rank Adaptation (LoRA) for efficiency. Defaults to False.
        free_rank (int, optional): Rank for LoRA decomposition. Defaults to 3.
        n_sample_params (int, optional): Number of parameter samples per epoch. Defaults to 5.
        n_sample_pref (int, optional): Number of preference samples per parameter. Defaults to 30.
        
    Returns:
        tuple: (hnet, psmodel) - Trained hypernetwork and Pareto Set model
    """
    n_params = problem.n_params
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    pu = problem.pu.to(device)
    pl = problem.pl.to(device)
    ideal_point = torch.tensor(problem.ideal_point).to(device)
    nadir_point = torch.tensor(problem.nadir_point).to(device)

    z = torch.zeros(n_obj).to(device)
    
    if lora_type: 
        hnet = PSModelLoRAHyper(
            n_params=n_params, 
            n_dim=n_dim, 
            n_obj=n_obj, 
            free_rank=free_rank,
            params_hidden_size=hpn_hidden_size,
            psm_hidden_size=psm_hidden_size, 
            psm_n_layer=psm_n_layer,
        )
        psmodel = PSModelLoRA(
            n_dim=n_dim, 
            n_obj=n_obj, 
            free_rank=free_rank, 
            hidden_size=psm_hidden_size, 
            n_layer=psm_n_layer,
        )
    else: 
        hnet = PSModelHyper(
            n_params=n_params, 
            n_dim=n_dim, 
            n_obj=n_obj, 
            params_hidden_size=hpn_hidden_size,
            psm_hidden_size=psm_hidden_size, 
            psm_n_layer=psm_n_layer
        )
        psmodel = PSModel(
            n_dim=n_dim, 
            n_obj=n_obj, 
            hidden_size=psm_hidden_size, 
            n_layer=psm_n_layer
        )
            
    logging.info(f"HyperNetwork size: {count_parameters(hnet)}.")
    if lora_type: logging.info(f"Use LoRA for PS model with r: {free_rank}.")

    hnet = hnet.to(device)
    psmodel = psmodel.to(device)
    
    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr)
    if lora_type: optimizer_baseModel = torch.optim.Adam(psmodel.base_model.parameters(), lr=1e-3)

    epoch_iter = trange(n_epochs)
    for epoch in epoch_iter: 
        hnet.train()
        if lora_type: psmodel.base_model.train()

        # sample n_sample_params parameters 
        params_sample = torch.rand([n_sample_params, n_params]).to(device) * (pu - pl) + pl
        
        optimizer.zero_grad() 
        for i, params in enumerate(params_sample):
            # obtain the weights for Pareto set model by the hypernetwork
            weights = hnet(params) 
            if lora_type: optimizer_baseModel.zero_grad()
            
            # sample n_pref_update preferences
            alpha = torch.ones(n_obj, device=device)
            pref = torch.distributions.Dirichlet(alpha).sample((n_sample_pref,))
            pref_vec = pref
            
            # get the predicted Pareto set 
            x = psmodel(pref_vec, weights) 
            
            # evaluate the loss 
            value = problem.evaluate(x, params)
            value = (value - ideal_point) / (nadir_point - ideal_point)

            # scalarize the objectives
            loss = loss_func(type=loss_type, 
                            preference_vector=pref_vec, 
                            func_value=value, 
                            z=z)

            # gradient descent
            loss.backward() 
            if lora_type: optimizer_baseModel.step()
        # hypernetwork update 
        optimizer.step()
            
        logging.info(f"Epoch: {epoch}, {loss_type} aggregation function with loss: {loss.item():.4f}.")

    return hnet, psmodel


def trainer_hpn_bbox(
        problem: any, 
        parameters: torch.tensor, 
        hnet: None, 
        psmodel: None, 
        hpn_hidden_size: int = 1024,
        psm_hidden_size: int = 128, 
        psm_n_layer: int = 2, 
        lr_hpn: float = 1e-5,
        lr_base: float = 1e-3,
        loss_type: str = 'stch', 
        device: torch.device = 'cpu',
        lora_type: bool = False, 
        free_rank: int = 3, 
        n_sample_pref: int = 30, 
        n_grad_esti: int = 5, 
):
    """
    Trains PPSL using black-box gradient estimation for non-differentiable problems.
    
    Args:
        problem (any): Parametric multi-objective optimization problem
        parameters (torch.tensor): Problem parameters to train on
        hnet (None): Existing hypernetwork (if None, creates new)
        psmodel (None): Existing PS model (if None, creates new)
        hpn_hidden_size (int): Hypernetwork hidden layer size
        psm_hidden_size (int): PS model hidden layer size
        psm_n_layer (int): Number of PS model layers
        lr_hpn (float): Learning rate for hypernetwork
        lr_base (float): Learning rate for PS base model (when using LoRA)
        loss_type (str): Scalarization function type
        device (torch.device): Computing device
        lora_type (bool): Whether to use LoRA adaptation
        free_rank (int): LoRA rank r
        n_sample_pref (int): Number of preference samples
        n_grad_esti (int): Number of samples for gradient estimation
        
    Returns:
        tuple: (hnet, psmodel) - Trained hypernetwork and PS model
        
    Notes:
        - Uses gradient estimation for black-box objectives (non-differentiable)
        - Gradient estimation samples n_grad_esti points for finite differences
    """
    n_params = problem.n_params
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    ideal_point = torch.tensor(problem.ideal_point).to(device)
    nadir_point = torch.tensor(problem.nadir_point).to(device)

    z = torch.zeros(n_obj).to(device)
    
    if hnet is None and psmodel is None: 
        if lora_type: 
            hnet = PSModelLoRAHyper(
                n_params=n_params, 
                n_dim=n_dim, 
                n_obj=n_obj, 
                free_rank=free_rank,
                params_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=psm_n_layer,
            )
            psmodel = PSModelLoRA(
                n_dim=n_dim, 
                n_obj=n_obj, 
                free_rank=free_rank, 
                hidden_size=psm_hidden_size, 
                n_layer=psm_n_layer,
            )
        else: 
            hnet = PSModelHyper(
                n_params=n_params, 
                n_dim=n_dim, 
                n_obj=n_obj, 
                params_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=psm_n_layer
            )
            psmodel = PSModel(
                n_dim=n_dim, 
                n_obj=n_obj, 
                hidden_size=psm_hidden_size, 
                n_layer=psm_n_layer
            )

    hnet = hnet.to(device)
    psmodel = psmodel.to(device)
    
    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr_hpn, weight_decay=1e-4)
    if lora_type: optimizer_baseModel = torch.optim.Adam(psmodel.base_model.parameters(), lr=lr_base, weight_decay=5e-3) # 

    hnet.train()
    if lora_type: psmodel.base_model.train()
    
    # transform the parameters to the true values
    params_sample = parameters.reshape(-1, n_params).to(device)

    hnet.train()
    optimizer.zero_grad() 
    # iterate the parameters
    for i, params in enumerate(params_sample):
        
        if lora_type: optimizer_baseModel.zero_grad()
        # obtain the weights for Pareto set model by the hypernetwork
        weights = hnet(params) 
        
        # sample n_pref_update preferences
        alpha = torch.ones(n_obj, device=device)
        pref = torch.distributions.Dirichlet(alpha).sample((n_sample_pref,))
        pref_vec = pref
        
        # get the predicted Pareto set 
        x = psmodel(pref_vec, weights) 
        
        # evaluate the gradient 
        grad = gradient_estimation(problem=problem, 
                                    x=x, 
                                    param=params, 
                                    loss_type=loss_type, 
                                    preference_vecotr=pref_vec, 
                                    z=z, 
                                    n_grad_esti=n_grad_esti,)

        x.backward(grad) 
        if lora_type: optimizer_baseModel.step()
            
    # hypernetwork update 
    optimizer.step()

    return hnet, psmodel


def trainer_ppsl_random(
        problem: any, 
        hpn_hidden_size: int,
        n_epochs: int, 
        psm_hidden_size: int = 256, 
        psm_n_layer: int = 2, 
        lr_hpn: float = 1e-5,
        lr_base: float = 1e-3,
        loss_type: str = 'stch', 
        device: torch.device = 'cpu',
        lora_type: bool = False, 
        free_rank: int = 3, 
        n_sample_params: int = 5, 
        n_sample_pref: int = 10, 
        verbose: bool = False, 
):
    """
    Trains PPSL with randomly sampled parameters from the parameter space.
    
    Args:
        problem (any): Parametric multi-objective optimization problem
        hpn_hidden_size (int): Hypernetwork hidden layer size
        n_epochs (int): Number of training epochs
        psm_hidden_size (int): PS model hidden layer size
        psm_n_layer (int): Number of PS model layers
        lr_hpn (float): Learning rate for hypernetwork
        lr_base (float): Learning rate for PS base model (LoRA)
        loss_type (str): Scalarization function type
        device (torch.device): Computing device
        lora_type (bool): Whether to use LoRA
        free_rank (int): LoRA rank r
        n_sample_params (int): Number of parameter samples per epoch
        n_sample_pref (int): Number of preference samples
        verbose (bool): Whether to print training progress
        
    Returns:
        tuple: (hnet, psmodel) - Trained hypernetwork and PS model
        
    Notes:
        - Parameters are uniformly sampled from [0,1]^n_params
    """
    n_params = problem.n_params
    hnet = None
    psmodel = None
    pu = problem.pu.to(device)
    pl = problem.pl.to(device)
    
    epoch_iter = trange(n_epochs)
    for epoch in epoch_iter: 
        # sample n_sample_params parameters 
        params_sample = torch.rand([n_sample_params, n_params]).to(device)
        
        hnet, psmodel = trainer_ppsl_fix_para(
            problem=problem, 
            parameters=params_sample, 
            hnet=hnet, 
            psmodel=psmodel,
            hpn_hidden_size=hpn_hidden_size, 
            psm_hidden_size=psm_hidden_size, 
            psm_n_layer=psm_n_layer, 
            lr_hpn=lr_hpn,
            lr_base=lr_base,
            loss_type=loss_type, 
            device=device, 
            lora_type=lora_type, 
            free_rank=free_rank, 
            n_sample_pref=n_sample_pref,
            verbose=verbose,
            )

    return hnet, psmodel


def trainer_ppsl_fix_para(
        problem: any, 
        parameters: torch.tensor, 
        hnet: None, 
        psmodel: None, 
        hpn_hidden_size: int,
        psm_hidden_size: int, 
        psm_n_layer: int, 
        lr_hpn: float = 1e-5,
        lr_base: float = 1e-3,
        loss_type: str = 'stch', 
        device: torch.device = 'cpu',
        lora_type: bool = False, 
        free_rank: int = 3, 
        n_sample_pref: int = 30, 
        verbose: bool = False,
):
    """
    Trains PPSL with fixed parameters for one optimization step.
    
    Args:
        problem (any): Parametric multi-objective optimization problem
        parameters (torch.tensor): Fixed parameters to train on
        hnet (None): Existing hypernetwork (if None, creates new)
        psmodel (None): Existing PS model (if None, creates new)
        hpn_hidden_size (int): Hypernetwork hidden layer size
        psm_hidden_size (int): PS model hidden layer size
        psm_n_layer (int): Number of PS model layers
        lr_hpn (float): Learning rate for hypernetwork
        lr_base (float): Learning rate for PS base model
        loss_type (str): Scalarization function type
        device (torch.device): Computing device
        lora_type (bool): Whether to use LoRA
        free_rank (int): LoRA rank r
        n_sample_pref (int): Number of preference samples
        verbose (bool): Whether to print training info
        
    Returns:
        tuple: (hnet, psmodel) - Updated hypernetwork and PS model
        
    Notes:
        - Preferences sampled from Dirichlet(α) where α = [1,...,1]
        - Objectives normalized using ideal and nadir points
        - Bilevel optimization: inner loop updates PS model, outer updates hypernetwork
    """
    # initialize the problems' characteristics
    n_params = problem.n_params
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    ideal_point = torch.tensor(problem.ideal_point).to(device)
    nadir_point = torch.tensor(problem.nadir_point).to(device)
    z = torch.zeros(n_obj).to(device)
    
    if hnet is None and psmodel is None: 
        if lora_type: 
            hnet = PSModelLoRAHyper(
                n_params=n_params, 
                n_dim=n_dim, 
                n_obj=n_obj, 
                free_rank=free_rank,
                params_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=psm_n_layer,
            )
            psmodel = PSModelLoRA(
                n_dim=n_dim, 
                n_obj=n_obj, 
                free_rank=free_rank, 
                hidden_size=psm_hidden_size, 
                n_layer=psm_n_layer,
            )
        else: 
            hnet = PSModelHyper(
                n_params=n_params, 
                n_dim=n_dim, 
                n_obj=n_obj, 
                params_hidden_size=hpn_hidden_size,
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=psm_n_layer
            )
            psmodel = PSModel(
                n_dim=n_dim, 
                n_obj=n_obj, 
                hidden_size=psm_hidden_size, 
                n_layer=psm_n_layer
            )
            
        if verbose: 
            logging.info(f"HyperNetwork size: {count_parameters(hnet)}.")
            if lora_type: logging.info(f"Use LoRA for PS model with r: {psmodel.free_rank}.")

    hnet = hnet.to(device)
    psmodel = psmodel.to(device)
    
    optimizer = torch.optim.Adam(hnet.parameters(), lr=lr_hpn, weight_decay=1e-2)
    if lora_type: optimizer_baseModel = torch.optim.Adam(psmodel.base_model.parameters(), lr=lr_base, weight_decay=1e-4)

    hnet.train()
    if lora_type: psmodel.base_model.train()

    # transform the parameters to the true values
    params_sample = parameters.reshape(-1, n_params).to(device)
    
    optimizer.zero_grad() 
    # iterate the parameters
    for i, params in enumerate(params_sample):
        
        # obtain the weights for Pareto set model by the hypernetwork
        weights = hnet(params) 
        if lora_type: optimizer_baseModel.zero_grad()
        
        # sample n_pref_update preferences
        alpha = torch.ones(n_obj, device=device)
        pref_vec = torch.distributions.Dirichlet(alpha).sample((n_sample_pref,))
        
        # get the predicted Pareto set 
        x = psmodel(pref_vec, weights) 
        
        # evaluate the loss 
        value = problem.evaluate(x, problem.unnormalize_p(params))
        value = (value - ideal_point) / (nadir_point - ideal_point)

        # scalarize the objectives
        loss = loss_func(type=loss_type, 
                        preference_vector=pref_vec, 
                        func_value=value, 
                        z=z)

        # gradient descent
        loss.backward() 
        if lora_type: optimizer_baseModel.step()
    # hypernetwork update 
    optimizer.step()
        
    if verbose: logging.info(f"{loss_type} aggregation function with loss: {loss.item():.4f}.")

    return hnet, psmodel


def trainer_psmodel(
        problem: any, 
        psm_hidden_size: int, 
        psm_n_layer: int, 
        n_epochs: int, 
        lr: float,
        loss_type: str, 
        device: torch.device,
        n_sample_pref: int = 30, 
):
    """
    Trains a Pareto Set model for non-parametric multi-objective optimization.
    
    Args:
        problem (any): Multi-objective optimization problem (non-parametric)
        psm_hidden_size (int): PS model hidden layer size
        psm_n_layer (int): Number of PS model layers
        n_epochs (int): Number of training epochs
        lr (float): Learning rate
        loss_type (str): Scalarization function type
        device (torch.device): Computing device
        n_sample_pref (int): Number of preference samples per iteration
        
    Returns:
        PSbaseModel: Trained PS model that maps preferences to Pareto optimal solutions
        
    Notes:
        - For non-parametric problems (no hypernetwork needed)
        - Preferences sampled from Dirichlet(α) where α = [1,...,1]
        - Objectives normalized to [0,1] using ideal and nadir points
    """
    n_dim = problem.n_dim
    n_obj = problem.n_obj
    ideal_point = torch.tensor(problem.ideal_point).to(device)
    nadir_point = torch.tensor(problem.nadir_point).to(device)
    z = torch.zeros(n_obj).to(device)
    
    psmodel = PSbaseModel(
        n_dim=n_dim, 
        n_obj=n_obj, 
        hidden_size=psm_hidden_size, 
        n_layer=psm_n_layer
    )

    psmodel = psmodel.to(device)
    optimizer = torch.optim.Adam(psmodel.parameters(), lr=lr)

    epoch_iter = trange(n_epochs)
    for epoch in epoch_iter: 
        psmodel.train()
            
        # sample n_pref_update preferences
        alpha = torch.ones(n_obj, device=device)
        pref = torch.distributions.Dirichlet(alpha).sample((n_sample_pref,))
        pref_vec = pref
        
        # get the predicted Pareto set 
        x = psmodel(pref_vec) 
        
        # evaluate the loss 
        value = problem.evaluate(x)
        value = (value - ideal_point) / (nadir_point - ideal_point)

        # scalarize the objectives
        loss = loss_func(type=loss_type, 
                        preference_vector=pref_vec, 
                        func_value=value, 
                        z=z)

        # gradient descent
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()

    return psmodel