import os
import time
import pickle
import argparse 

import numpy as np
import torch

from pymoo.algorithms.moo.dnsga2 import DNSGA2
from pymoo.algorithms.moo.kgb import KGB
from pymoo.core.callback import CallbackCollection, Callback
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.problems.dyn import TimeSimulation
from pymoo.termination import get_termination
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import Hypervolume

from problems.problem import mop_dyn
from trainer import trainer_ppsl_fix_para, generate_ps, trainer_hpn_bbox

import itertools
from datetime import datetime


def run_dmop(
        method: str, 
        problem_string: str, 
        tau_t: int = 5, 
        n_t: int = 10, 
        n_var: int = 10, 
        pop_size: int = 200, 
        repetition: int = 10, 
        count_metric: bool = True, 
        save: bool = True, 
        verbose: bool = True, 
        seed: int = None, 
        device: str = 'cpu',
        suffix: str = '', 
        n_grad_esti: int = 5, 
        n_change_time: int = 50, 
): 
    if suffix != '': suffix = '_'+suffix
    # Experimental Settings
    change_frequency = tau_t
    change_severity = n_t
    max_n_gen = n_change_time * change_frequency
    termination = get_termination("n_gen", max_n_gen)

    def reset_metrics():
        global po_gen, igds, hvs, hvds, igd_all, hv_all
        po_gen = []
        igds = []
        hvs = []
        hvds = []
        igd_all = []
        hv_all = []

    def update_metrics(algorithm):
        global ind
        ind += 1
        _F = algorithm.opt.get("F")
        if count_metric:
            PF = algorithm.problem._calc_pareto_front(n_pareto_points=pop_size)
            igd = IGD(PF).do(_F)
            igd_all.append(igd)
            hv = Hypervolume(pf=PF).do(_F)
            hv_all.append(hv)

            if (ind % tau_t == 0): 
                igds.append(igd)
                hvs.append(hv)
                # hvds.append(
                #     Hypervolume().do(PF) - Hypervolume().do(_F)
                # )
                po_gen.append(algorithm.opt.get("X"))

    class DefaultDynCallback(Callback):

        def _update(self, algorithm):

            update_metrics(algorithm)

    # Function to run an algorithm and return the results
    def run_algorithm(problem, algorithm, termination, seed):
        reset_metrics()
        simulation = TimeSimulation()
        callback = CallbackCollection(DefaultDynCallback(), simulation)
        res = minimize(problem, algorithm, termination=termination, callback=callback, seed=seed, verbose=False)
        return res, igds, hvs, hvds, po_gen, igd_all, hv_all
    
    if count_metric: 
        POS = []
        IGDS = []
        HVS = []
        IGD_ALL = []
        HV_ALL = []
        TIMES = []
        HVDS = []
        IGDS_LS = [] 
        HVS_LS = []

    file_path = f"results/dmop/{problem_string.lower()}/{method.lower()}_nt_{n_t}_taut_{tau_t}{suffix}.pickle"
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    match method.lower(): 
        case "dnsga2a": 
            # DNSGA2A
            for i in range(repetition): 
                problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
                algorithm = DNSGA2(pop_size=pop_size, version="A")
                start = time.time()
                res, igds_tmp, hvs_tmp, hvds_tmp, po_gen_tmp, igd_all_tmp, hv_all_tmp = run_algorithm(problem, algorithm, termination, seed)
                time_tmp = time.time() - start
                IGDS.append(igds_tmp)
                HVS.append(hvs_tmp)
                POS.append(po_gen_tmp)
                TIMES.append(time_tmp)
                HVDS.append(hvds_tmp)
                IGD_ALL.append(igd_all_tmp)
                HV_ALL.append(hv_all_tmp)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {i}: MIGD: {np.mean(igds_tmp):.2e}, MHV: {np.mean(hvs_tmp):.2e}, Evaluation Time: {time_tmp:.2f}")
        
        case "dnsga2b": 
            # DNSGA2B
            for i in range(repetition): 
                problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
                algorithm = DNSGA2(pop_size=pop_size, version="B")
                start = time.time()
                res, igds_tmp, hvs_tmp, hvds_tmp, po_gen_tmp, igd_all_tmp, hv_all_tmp = run_algorithm(problem, algorithm, termination, seed)
                time_tmp = time.time() - start
                IGDS.append(igds_tmp)
                HVS.append(hvs_tmp)
                POS.append(po_gen_tmp)
                TIMES.append(time_tmp)
                HVDS.append(hvds_tmp)
                IGD_ALL.append(igd_all_tmp)
                HV_ALL.append(hv_all_tmp)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {i}: MIGD: {np.mean(igds_tmp):.2e}, MHV: {np.mean(hvs_tmp):.2e}, Evaluation Time: {time_tmp:.2f}")
            
        case "kgb": 
            # KGB
            for i in range(repetition): 
                problem = get_problem(problem_string, taut=change_frequency, nt=change_severity, n_var=n_var)
                algorithm = KGB(pop_size=pop_size)
                start = time.time()
                res, igds_tmp, hvs_tmp, hvds_tmp, po_gen_tmp, igd_all_tmp, hv_all_tmp = run_algorithm(problem, algorithm, termination, seed)
                time_tmp = time.time() - start
                IGDS.append(igds_tmp)
                HVS.append(hvs_tmp)
                POS.append(po_gen_tmp)
                TIMES.append(time_tmp)
                HVDS.append(hvds_tmp)
                IGD_ALL.append(igd_all_tmp)
                HV_ALL.append(hv_all_tmp)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {i}: MIGD: {np.mean(igds_tmp):.2e}, MHV: {np.mean(hvs_tmp):.2e}, Evaluation Time: {time_tmp:.2f}")

        case "ppsl": 
            # PPSL
            for rep in range(repetition): 
                start = time.time()
                reset_metrics()
                igds_ls = [] 
                hvs_ls = []
                times = []
                true_pf = []
                pred_pf = []
                # experiment with generation index
                p = mop_dyn(pname=problem_string, n_dim=n_var, taut=change_frequency, nt=change_severity)

                hnet, psmodel = None, None
                ind = 1
                for i in range(max_n_gen): 
                    params = torch.ones((int(pop_size / 5), 1)) * i / (max_n_gen)
                    hnet, psmodel = trainer_ppsl_fix_para(
                        problem=p, 
                        parameters=params, 
                        hnet=hnet, 
                        psmodel=psmodel, 
                        hpn_hidden_size=1024, 
                        psm_hidden_size=256, 
                        psm_n_layer=2, 
                        lr_hpn=1e-5, 
                        lr_base=1e-3, 
                        loss_type='stch', 
                        lora_type=True, 
                        free_rank=3, 
                        n_sample_pref=5, 
                        device=device
                    )
                    
                    pred_ps = generate_ps(
                        problem=p, 
                        param=torch.tensor([i / max_n_gen], device=device),  
                        hypernet=hnet, 
                        psmodel=psmodel, 
                        n_samples=pop_size, 
                        device=device, 
                    )
                    
                    pred_ps_ls = generate_ps(
                        problem=p, 
                        param=torch.tensor([i / max_n_gen], device=device),  
                        hypernet=hnet, 
                        psmodel=psmodel, 
                        n_samples=2000, 
                        device=device, 
                    )

                    ind += 1
                    if count_metric:
                        _F = p.evaluate(pred_ps).cpu().numpy()
                        PF = p._calc_pareto_front(n_pareto_points=pop_size)
                        igd = IGD(PF).do(_F)
                        hv = Hypervolume(pf=PF).do(_F)
                        igd_all.append(igd)
                        hv_all.append(hv)
                        if (ind % tau_t == 0):
                            igds.append(igd)
                            hvs.append(hv)
                            _F_ls = p.evaluate(pred_ps_ls).cpu().numpy()
                            igd_ls = IGD(PF).do(_F_ls)
                            hv_ls = Hypervolume(pf=PF).do(_F_ls)
                            igds_ls.append(igd_ls) 
                            hvs_ls.append(hv_ls)
                            po_gen.append(_F)
                            # report for draw of PF 
                            times.append(p.time)
                            true_pf.append(PF)
                            pred_pf.append(_F)
                            if verbose: print(f"Epoch {ind}: PARAM: {params[0].item():.2e}, IGD(200): {igd:.2e}, IGD(2000): {igd_ls:.2e}, HV(200): {hv:.2e}")
                    
                    p.tic()

                time_tmp = time.time() - start
                IGDS.append(igds)
                HVS.append(hvs)
                POS.append(po_gen)
                TIMES.append(time_tmp)
                HVDS.append(hvds)
                IGD_ALL.append(igd_all)
                HV_ALL.append(hv_all)
                IGDS_LS.append(igds_ls)
                HVS_LS.append(hvs_ls)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {rep}: MIGD(200): {np.mean(igds):.2e}, MIGD(2000): {np.mean(igds_ls):.2e}, MHV(200): {np.mean(hvs):.2e}, Evaluation Time: {time_tmp:.2f}")
                    
        case "ppsl_bbox": 
            # PPSL (Black Box)
            for rep in range(repetition): 
                start = time.time()
                reset_metrics()
                igds_ls = [] 
                hvs_ls = []
                # experiment with generation index
                p = mop_dyn(pname=problem_string, n_dim=n_var, taut=change_frequency, nt=change_severity)

                hnet, psmodel = None, None
                ind = 1
                for i in range(max_n_gen): 
                    params = torch.ones((int(pop_size / (n_grad_esti*5)), 1)) * i / (max_n_gen)
                    hnet, psmodel = trainer_hpn_bbox(
                        problem=p, 
                        parameters=params, 
                        hnet=hnet, 
                        psmodel=psmodel, 
                        hpn_hidden_size=1024, 
                        psm_hidden_size=256, 
                        psm_n_layer=3, 
                        lr_hpn=3e-5, 
                        lr_base=8e-4, 
                        loss_type='stch', 
                        lora_type=True, 
                        free_rank=3, 
                        n_sample_pref=5, 
                        n_grad_esti=n_grad_esti,
                    )
                    
                    pred_ps = generate_ps(
                        problem=p, 
                        param=torch.tensor([i / max_n_gen]),  
                        hypernet=hnet, 
                        psmodel=psmodel, 
                        n_samples=1000, 
                        device=device, 
                    )

                    ind += 1

                    if count_metric:
                        _F = p.evaluate(pred_ps).cpu().numpy()
                        PF = p._calc_pareto_front(n_pareto_points=pop_size)
                        igd = IGD(PF).do(_F)
                        hv = Hypervolume(pf=PF).do(_F)
                        igd_all.append(igd)
                        hv_all.append(hv)
                        if (ind % tau_t == 0):
                            igds.append(igd)
                            hvs.append(hv)
                            # hvds.append(
                            #     Hypervolume().do(PF) - Hypervolume().do(_F)
                            # )
                            print(f"Epoch {i}: IGD: {igd:.2e}, MHV: {hv:.2e}")
                    
                    p.tic()

                time_tmp = time.time() - start
                IGDS.append(igds)
                HVS.append(hvs)
                POS.append(po_gen)
                TIMES.append(time_tmp)
                HVDS.append(hvds)
                IGD_ALL.append(igd_all)
                HV_ALL.append(hv_all)
                IGDS_LS.append(igds_ls)
                HVS_LS.append(hvs_ls)
                if verbose: 
                    print(f"Method: {method} | Problem: {problem_string}, tau_t: {tau_t}, n_t: {n_t} | Trial {rep}: MIGD: {np.mean(igds):.2e}, MHV: {np.mean(hvs):.2e}, Evaluation Time: {time_tmp:.2f}")

    if save: 
        with open(file_path, 'wb') as f: 
            pickle.dump(
                {"igds": IGDS, 
                "hvs": HVS, 
                "hvds": HVDS, 
                "pos": POS, 
                "times": TIMES,
                "igd_all": IGD_ALL, 
                "hv_all": HV_ALL,
                "igds_ls": IGDS_LS, 
                "hvs_ls": HVS_LS,
                }, f
            )
    else: 
        with open(file_path, 'wb') as f: 
            pickle.dump(
                 {"t": times, 
                "pf": true_pf, 
                "pred_pf": pred_pf, 
                }, f
            )


def extra_experiments():
    # params
    tau_t_values = [20, 40]
    n_t_values = [10, 20]
    
    # pop_size
    problems_config = {
        **{f"df{i}": 100 for i in range(1, 10)},  # df1-df9: pop_size=100
        **{f"df{i}": 150 for i in range(10, 15)}  # df10-df14: pop_size=150
    }
    
    total_experiments = len(tau_t_values) * len(n_t_values) * len(problems_config)
    print(f"Total experiments to run: {total_experiments}")
    print(f"Starting at {datetime.now()}")
    
    count = 0
    for tau_t, n_t in itertools.product(tau_t_values, n_t_values):
        for problem, pop_size in problems_config.items():
            count += 1
            print(f"\n[{count}/{total_experiments}] Running: {problem} (tau_t={tau_t}, n_t={n_t}, pop_size={pop_size})")
            
            try:
                run_dmop(
                    method='ppsl',
                    problem_string=problem,
                    tau_t=tau_t,
                    n_t=n_t,
                    pop_size=pop_size,
                    repetition=20,
                    n_change_time=100,
                    device='cuda',
                    verbose=True,
                    save=True
                )
                print(f"✓ Completed: {problem}")
            except Exception as e:
                print(f"✗ Failed: {problem}, Error: {e}")
    
    print(f"\nAll experiments completed at {datetime.now()}")



if __name__ == "__main__":
    print()
    # extra_experiments()