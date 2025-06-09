import os
import numpy as np 
import torch 

import logging 
import time
import pickle

from pymoo.indicators.hv import HV
from pymoo.core.problem import Problem 
from pymoo.util.ref_dirs import get_reference_directions
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.optimize import minimize

from problems import mop_sc
from trainer import trainer_ppsl_random, trainer_psmodel, generate_ps

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class pmop_pymoo(Problem): 
    def __init__(self, problem, param):
        self.p = problem 
        self.param = param
        super().__init__(n_var=self.p.n_dim, n_obj=self.p.n_obj, n_ieq_constr=0, xl=np.zeros(self.p.n_dim), xu=np.ones(self.p.n_dim))

    def _evaluate(self, x, out, *args, **kwargs):
        out["F"] = self.p.evaluate(torch.tensor(x).to(device), self.param).cpu().numpy()

class pmop_psl(): 
    def __init__(self, problem, param):
        self.p = problem 
        self.n_dim = self.p.n_dim
        self.n_obj = self.p.n_obj 
        self.ideal_point = self.p.ideal_point
        self.nadir_point = self.p.nadir_point
        self.param = param  
    
    def evaluate(self, x): 
        return self.p.evaluate(x, self.param)

def run_mopsc(
    problem_name: str, 
    shared_comp: list, 
    hpn_hidden_size: int,
    psm_hidden_size: int, 
    n_hidden_layer: int, 
    free_rank: int, 
    loss_type: str, 
    n_repetition: int, 
    run_ppsl: bool, 
    run_noLora: bool, 
    run_moea: bool, 
    run_psl: bool, 
    n_sample_params_test: int = 10, 
    lr_hpn: float = 5e-5, 
    lr_base: float = 50e-5, 
    save_name: str = None, 
    device: str = 'cuda', 
): 
    print(f"---------------Problem {problem_name}--------------------")
    res = {}
    ## define the problem
    p = mop_sc(pname=problem_name, share_comp=shared_comp) 

    file_path = f"results/mopsc/{problem_name.lower()}/test_cases_{''.join(map(str, shared_comp))}.pickle"
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            params = pickle.load(f)
    else:
        params = torch.rand((n_sample_params_test, p.n_params)).to(device) * (p.pu.to(device)-p.pl.to(device)) + p.pl.to(device)
        
        os.makedirs(os.path.dirname(file_path), exist_ok=True) 
        with open(file_path, 'wb') as f:
            pickle.dump(params, f)
 
    hv = HV(ref_point=np.array([1.1] * p.n_obj))
    
    ## PPSL
    if run_ppsl: 
        n_sample_params = 5 if p.n_obj == 2 else 8
        t0_train = 0 
        t0_infer_s, t0_infer_l = 0, 0
        hv_ppsl_s, hv_ppsl_l = [], []
        print(f"----------------------PPSL------------------------")
        for i in range(n_repetition): 
            t0_start = time.time()
            hpnet, psmodel = trainer_ppsl_random(
                problem=p, 
                hpn_hidden_size=hpn_hidden_size, 
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=n_hidden_layer, 
                n_epochs=1000, 
                lr_base=lr_base, 
                lr_hpn=lr_hpn, 
                loss_type=loss_type, 
                device=device, 
                n_sample_pref=30,
                n_sample_params=n_sample_params,
                lora_type=True, 
                free_rank=free_rank,
            )
            t0_end = time.time()
            t0_train += (t0_end - t0_start) 
            
            hv_value_s, hv_value_l = [], []
            # calculate and report the hypervolume value
            for param in params: 
                hpnet.eval()
                generated_ps, generated_pf = [], []
                
                t0_infer_start = time.time()
                generated_ps_s = generate_ps(p, p.normalize_p(param), hpnet, psmodel, 1000, device)
                t0_infer_end = time.time()
                t0_infer_s += (t0_infer_end - t0_infer_start)
                t0_infer_start = time.time()
                generated_ps_l = generate_ps(p, p.normalize_p(param), hpnet, psmodel, 2000, device)
                t0_infer_end = time.time()
                t0_infer_l += (t0_infer_end - t0_infer_start)
            
                obj_s = p.evaluate(generated_ps_s, param).cpu().numpy()
                obj_l = p.evaluate(generated_ps_l, param).cpu().numpy()
                
                generated_pf_s = (obj_s - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                generated_pf_l = (obj_l - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                
                hv_value_s.append(hv(generated_pf_s))
                hv_value_l.append(hv(generated_pf_l))
            
                print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV (small): {hv_value_s[-1]:.4f}, HV (large): {hv_value_l[-1]:.4f}.")
                
            hv_ppsl_s.append(hv_value_s)
            hv_ppsl_l.append(hv_value_l)
        
        res['ppsl_hv_small'], res['ppsl_hv_large'] = hv_ppsl_s, hv_ppsl_l
        res['ppsl_training_time'] = t0_train / n_repetition
        res['ppsl_small_inference_time'] = t0_infer_s / n_repetition 
        res['ppsl_large_inference_time'] = t0_infer_l / n_repetition
    

    ## no LoRA 
    if run_noLora:
        n_sample_params = 5 if p.n_obj == 2 else 8
        t1_train = 0 
        t1_infer = 0
        hv_ppsl_nolora = []
        print(f"----------------------PPSL (no LoRA)------------------------")
        for i in range(n_repetition): 
            t1_start = time.time()
            hpnet, psmodel = trainer_ppsl_random(
                problem=p, 
                hpn_hidden_size=hpn_hidden_size, 
                psm_hidden_size=psm_hidden_size, 
                psm_n_layer=n_hidden_layer, 
                n_epochs=1000, 
                lr_hpn=lr_hpn, 
                loss_type=loss_type, 
                device=device, 
                n_sample_pref=10,
                n_sample_params=n_sample_params,
                lora_type=False, 
                free_rank=free_rank,
            )
            t1_end = time.time()
            t1_train += (t1_end - t1_start) 
            
            hv_value_nolora = []
            # calculate and report the hypervolume value
            for param in params: 
                hpnet.eval()
                generated_ps, generated_pf = [], []
                
                t1_infer_start = time.time()
                generated_ps = generate_ps(p, param, hpnet, psmodel, 1000, device)
                t1_infer_end = time.time()
                t1_infer += (t1_infer_end - t1_infer_start)
            
                obj = p.evaluate(generated_ps, param).cpu().numpy()
                
                generated_pf = (obj - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                
                hv_value_nolora.append(hv(generated_pf))
            
                print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV: {hv_value_nolora[-1]:.4f}.")
                
            hv_ppsl_nolora.append(hv_value_nolora)
        
        res['ppsl_nolora_hv'] = hv_ppsl_nolora
        res['ppsl_nolora_training_time'] = t1_train / n_repetition
        res['ppsl_nolora_inference_time'] = t1_infer / n_repetition 
    

    ## PSL
    if run_psl: 
        print(f"----------------------PSL------------------------")
        t_psl_train, t_psl_infer = 0., 0.
        hv_psl = [] 
        for i in range(n_repetition): 
            hv_value = []
            for param in params: 
                p_psl = pmop_psl(p, param)
                t0_start = time.time()
                psl_model = trainer_psmodel(
                    problem=p_psl, 
                    psm_hidden_size=1024, 
                    psm_n_layer=2, 
                    n_epochs=2000, 
                    lr=1e-3, 
                    loss_type='stch', 
                    device=device, 
                    n_sample_pref=10,
                )
                t0_end = time.time()
                t_psl_train += (t0_end - t0_start) 
                
                # calculate and report the hypervolume value
                psl_model.eval()
                generated_ps, generated_pf = [], []
                
                t0_infer_start = time.time()
                with torch.no_grad():
                    alpha = torch.ones(p.n_obj, device=device)
                    pref = torch.distributions.Dirichlet(alpha).sample((1000,))
                    generated_ps = psl_model(pref)
                t0_infer_end = time.time()
                t_psl_infer += (t0_infer_end - t0_infer_start)
            
                obj = p.evaluate(generated_ps, param).cpu().numpy()
                
                generated_pf = (obj - p.ideal_point) / (p.nadir_point - p.ideal_point) 
                
                hv_value.append(hv(generated_pf))
            
                print(f"Repetition {i} --> Para: {param.round(decimals=1)}, HV: {hv_value[-1]:.4f}.")
                
            hv_psl.append(hv_value)
    
        res['psl_hv'] = hv_psl
        res['psl_training_time'] = t_psl_train / n_repetition
        res['psl_inference_time'] = t_psl_infer / n_repetition 
    

    ## MOEAs
    if run_moea:
        print(f"----------------------MOEAs------------------------")
        n_evals = 41000
        if p.n_obj == 2: 
            n_partitions = 99
            n_evals = 26000
        elif p.n_obj == 3: 
            n_partitions = 13
            
        ref_dirs = get_reference_directions("uniform", p.n_obj, n_partitions=n_partitions)
        
        hv_moead, hv_nsga2, hv_nsga3 = [], [], []
        t_moead, t_nsga2, t_nsga3 = 0., 0., 0.
        for i in range(n_repetition): 
            hv_value_moead, hv_value_nsga2, hv_value_nsga3 = [], [], []
            for param in params: 
                p_pymoo = pmop_pymoo(p, param)
                # MOEA/D
                t_moead_start = time.time()
                algorithm = MOEAD(ref_dirs, n_neighbors=30, prob_neighbor_mating=0.8)
                opti_moead = minimize(p_pymoo, algorithm, termination=("n_evals", n_evals), verbose=False)
                t_moead_end = time.time()
                t_moead += (t_moead_end - t_moead_start)
                # NSGA-2
                t_nsga2_start = time.time()
                algorithm = NSGA2(pop_size=100)
                opti_nsga2 = minimize(p_pymoo, algorithm, termination=("n_evals", n_evals), verbose=False)
                t_nsga2_end = time.time()
                t_nsga2 += (t_nsga2_end - t_nsga2_start)
                # NSGA-3
                t_nsga3_start = time.time()
                algorithm = NSGA3(pop_size=100, ref_dirs=ref_dirs)
                opti_nsga3 = minimize(p_pymoo, algorithm, termination=("n_evals", n_evals), verbose=False)
                t_nsga3_end = time.time()
                t_nsga3 += (t_nsga3_end - t_nsga3_start)
                
                moead_pf = (opti_moead.F - p.ideal_point) / (p.nadir_point - p.ideal_point)
                nsga2_pf = (opti_nsga2.F - p.ideal_point) / (p.nadir_point - p.ideal_point)
                nsga3_pf = (opti_nsga3.F - p.ideal_point) / (p.nadir_point - p.ideal_point)
                hv_value_moead.append(hv(moead_pf))
                hv_value_nsga2.append(hv(nsga2_pf))
                hv_value_nsga3.append(hv(nsga3_pf))
                print(f"Repetition {i} **MOEA/D** --> Para: {param.round(decimals=1)}, HV: {hv_value_moead[-1]:.4f}.")
                print(f"Repetition {i} **NSGA-2** --> Para: {param.round(decimals=1)}, HV: {hv_value_nsga2[-1]:.4f}.")
                print(f"Repetition {i} **NSGA-3** --> Para: {param.round(decimals=1)}, HV: {hv_value_nsga3[-1]:.4f}.")
                
            hv_moead.append(hv_value_moead)
            hv_nsga2.append(hv_value_nsga2)
            hv_nsga3.append(hv_value_nsga3)
        
        res['moead_hv'] = hv_moead
        res['moead_time'] = t_moead / n_repetition
        res['nsga2_hv'] = hv_nsga2
        res['nsga2_time'] = t_nsga2 / n_repetition
        res['nsga3_hv'] = hv_nsga3
        res['nsga3_time'] = t_nsga3 / n_repetition
    
    
    
    ## save results
    if save_name is not None: 
        os.makedirs(os.path.dirname(save_name), exist_ok=True)
        with open(save_name+'.pickle', 'wb') as f: 
            pickle.dump(res, f)



if __name__ == "__main__": 
    print() 
    problem = ['RE21', 'RE37', 'RE33', 'RE36']
    share = [[0],[1],[2],[3],[0,1],[1,2],[2,3],[0,1,2],[1,2,3]] 
    # psm_sizes = [128, 256, 512]
    # n_layers = [2, 3, 4]
    # rank_sizes = [2, 3, 5]

    for a in problem: 
        for b in share: 
            save_name = f"./results/mopsc/{a.lower()}/{'ppsl'}_shared_{''.join(map(str, b))}"
            # save_name = f"./results/mopsc/previous/{'LoRAorNot'}_{a}_{''.join(map(str, b))}_hpn1024_psm512layer1_r{3}_stch"
            run_mopsc(
                problem_name=a, 
                shared_comp=b, 
                hpn_hidden_size=1024, 
                psm_hidden_size=512, 
                n_hidden_layer=2, 
                loss_type='stch', 
                n_repetition=3, 
                run_moea=False, 
                run_psl=False, 
                run_ppsl=True,
                run_noLora=False, 
                n_sample_params_test=10, 
                lr_hpn=1e-5, 
                lr_base=1e-3, 
                free_rank=3, 
                save_name=save_name, 
                device='cuda',
            )